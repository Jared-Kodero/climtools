#!/usr/bin/env python3
# ruff: noqa: UP006, UP035, UP045
"""Build the XGEO environment and parallel NetCDF-4 Python stack.

Run this file with a system Python. If no non-base conda environment or
virtualenv is active, it creates/updates a conda environment from boot.yaml.
The script then locates an HPC MPI/netcdf-mpi module pair or builds parallel
HDF5 and netCDF-C from source, builds mpi4py and netCDF4 against that stack,
applies environment.yaml for newly managed conda environments, and installs
XGEO editable.

Expected location: <repository>/env/setup_env.py
/usr/bin/python3 env/setup_env.py XGEO
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HDF5_SOURCE_VERSION = "1.14.5"
NETCDF_C_SOURCE_VERSION = "4.9.3"
MINICONDA_URL = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

PARALLEL_CHECK = """\
import sys
try:
    import mpi4py.MPI  # noqa: F401
    import netCDF4
except ImportError:
    sys.exit(1)
sys.exit(0 if netCDF4.__has_parallel4_support__ else 1)
"""

PARALLEL_C_TEST = """\
#include <netcdf_meta.h>
#if !defined(NC_HAS_PARALLEL4) || !NC_HAS_PARALLEL4
#error "no parallel netcdf4"
#endif
int main(void) { return 0; }
"""


MODULE_INIT = r"""if ! type module >/dev/null 2>&1; then
    if [[ -n "${MODULESHOME:-}" && -f "${MODULESHOME}/init/bash" ]]; then
        source "${MODULESHOME}/init/bash"
    elif [[ -f /etc/profile.d/modules.sh ]]; then
        source /etc/profile.d/modules.sh
    elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
        source /usr/share/lmod/lmod/init/bash
    fi
fi
"""


PHASE_COUNT = 0


def phase(text: str) -> None:
    global PHASE_COUNT

    PHASE_COUNT += 1
    n_cols = shutil.get_terminal_size(fallback=(80, 24)).columns

    print(f"\n[{PHASE_COUNT}] {text}")
    print("-" * n_cols)


class SetupError(RuntimeError):
    """Fatal XGEO environment setup error."""


def run(
    command: List[str],
    *,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[Path] = None,
    capture_output: bool = False,
    check: bool = True,
    input_text: Optional[str] = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        check=check,
        cwd=cwd,
        env=env,
        text=True,
        input=input_text,
        capture_output=capture_output,
    )


def download(url: str, destination: Path) -> None:
    with (
        urllib.request.urlopen(url, timeout=60) as response,
        destination.open("wb") as output,
    ):
        shutil.copyfileobj(response, output)


def extract_tar(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(destination)


def prepend_path(env: Dict[str, str], directory: Path) -> None:
    old_path = env.get("PATH", "")
    env["PATH"] = f"{directory}{os.pathsep}{old_path}" if old_path else str(directory)


def prepend_library_path(env: Dict[str, str], directory: Path) -> None:
    old_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        f"{directory}{os.pathsep}{old_path}" if old_path else str(directory)
    )


def executable_in_env(name: str, env: Dict[str, str]) -> Optional[str]:
    return shutil.which(name, path=env.get("PATH"))


def conda_executable() -> Optional[Path]:
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).is_file():
        return Path(conda_exe).resolve()

    conda = shutil.which("conda")
    return Path(conda).resolve() if conda else None


def install_miniconda() -> Path:
    prefix = Path.home() / "miniconda3"
    installer = prefix / "miniconda.sh"
    conda = prefix / "bin" / "conda"

    print("installing Miniconda")
    prefix.mkdir(parents=True, exist_ok=True)
    download(MINICONDA_URL, installer)
    try:
        run(["bash", str(installer), "-b", "-u", "-p", str(prefix)])
    finally:
        installer.unlink(missing_ok=True)

    if not conda.is_file():
        raise SetupError(
            f"Miniconda installation completed but conda was not found at {conda}"
        )
    return conda


def conda_envs(conda: Path) -> Dict[str, Path]:
    result = run(
        [str(conda), "env", "list", "--json"],
        capture_output=True,
    )
    data = json.loads(result.stdout)
    envs: Dict[str, Path] = {}
    for raw_prefix in data.get("envs", []):
        prefix = Path(raw_prefix)
        envs[prefix.name] = prefix
    return envs


def conda_prefix_for_name(conda: Path, env_name: str) -> Path:
    envs = conda_envs(conda)
    if env_name not in envs:
        raise SetupError(
            f"Conda environment '{env_name}' was created but its prefix was not found"
        )
    return envs[env_name]


def shell_has_module(base_env: Dict[str, str]) -> bool:
    command = f"{MODULE_INIT}\ntype module >/dev/null 2>&1"
    result = subprocess.run(
        ["bash", "-c", command],
        check=False,
        env=base_env,
    )
    return result.returncode == 0


def module_names(query: str, base_env: Dict[str, str]) -> List[str]:
    command = f"{MODULE_INIT}\nmodule -t spider {shlex.quote(query)} 2>&1 || true"
    result = run(["bash", "-c", command], env=base_env, capture_output=True)
    pattern = re.compile(rf"{re.escape(query)}/[^\s]+")
    names = {
        match.group(0).rstrip(" ,:;)") for match in pattern.finditer(result.stdout)
    }

    def version_key(name: str) -> Tuple[Tuple[int, object], ...]:
        parts = re.findall(r"\d+|\D+", name.lower())
        return tuple((0, int(part)) if part.isdigit() else (1, part) for part in parts)

    return sorted(names, key=version_key, reverse=True)


def environment_after_module_load(
    mpi_module: str,
    netcdf_module: str,
    base_env: Dict[str, str],
) -> Optional[Dict[str, str]]:
    command = "\n".join(
        [
            MODULE_INIT,
            "module purge >/dev/null 2>&1 || true",
            f"module load {shlex.quote(mpi_module)} >/dev/null 2>&1",
            f"module load {shlex.quote(netcdf_module)} >/dev/null 2>&1",
            "env -0",
        ]
    )
    result = subprocess.run(
        ["bash", "-c", command],
        check=False,
        env=base_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return None

    loaded = dict(base_env)
    for item in result.stdout.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        loaded[key.decode(errors="surrogateescape")] = value.decode(
            errors="surrogateescape"
        )
    loaded_path = loaded.get("PATH", "")
    base_path = base_env.get("PATH", "")
    if base_path and base_path not in loaded_path:
        loaded["PATH"] = (
            f"{loaded_path}{os.pathsep}{base_path}" if loaded_path else base_path
        )
    return loaded


def parallel_stack_usable(env: Dict[str, str]) -> bool:
    mpicc = executable_in_env("mpicc", env)
    nc_config = executable_in_env("nc-config", env)
    if not mpicc or not nc_config:
        return False

    cflags_result = run(
        [nc_config, "--cflags"],
        env=env,
        capture_output=True,
        check=False,
    )
    cflags = shlex.split(cflags_result.stdout) if cflags_result.returncode == 0 else []
    compile_result = run(
        [mpicc, *cflags, "-fsyntax-only", "-x", "c", "-"],
        env=env,
        check=False,
        input_text=PARALLEL_C_TEST,
        capture_output=True,
    )
    return compile_result.returncode == 0


def find_module_stack(
    base_env: Dict[str, str],
) -> Optional[Tuple[Dict[str, str], str, str]]:
    if not shell_has_module(base_env):
        return None

    mpi_candidates = module_names("hpcx-mpi", base_env)
    netcdf_candidates = module_names("netcdf-mpi", base_env)
    if not mpi_candidates or not netcdf_candidates:
        return None

    for netcdf_module in netcdf_candidates:
        for mpi_module in mpi_candidates:
            print(f"trying module stack: {mpi_module} + {netcdf_module}")
            loaded_env = environment_after_module_load(
                mpi_module,
                netcdf_module,
                base_env,
            )
            if loaded_env is None or not parallel_stack_usable(loaded_env):
                continue

            nc_config = executable_in_env("nc-config", loaded_env)
            mpicc = executable_in_env("mpicc", loaded_env)
            if nc_config is None or mpicc is None:
                continue
            prefix_result = run(
                [nc_config, "--prefix"],
                env=loaded_env,
                capture_output=True,
            )
            loaded_env["NETCDF4_DIR"] = prefix_result.stdout.strip()
            loaded_env["CC"] = mpicc
            print(f"using module stack: {mpi_module} + {netcdf_module}")
            return loaded_env, mpi_module, netcdf_module
    return None


def build_source_stack(
    prefix: Path,
    work_dir: Path,
    env: Dict[str, str],
) -> Dict[str, str]:
    mpicc = executable_in_env("mpicc", env)
    if mpicc is None:
        raise SetupError(
            "No MPI compiler (mpicc) found. Load or install an MPI implementation first."
        )

    prefix.mkdir(parents=True, exist_ok=True)

    print(f"building HDF5 {HDF5_SOURCE_VERSION} (parallel) into {prefix}")
    hdf5_archive = work_dir / "hdf5.tar.gz"
    download(
        (
            "https://github.com/HDFGroup/hdf5/releases/download/"
            f"hdf5_{HDF5_SOURCE_VERSION}/hdf5-{HDF5_SOURCE_VERSION}.tar.gz"
        ),
        hdf5_archive,
    )
    extract_tar(hdf5_archive, work_dir)
    hdf5_src = work_dir / f"hdf5-{HDF5_SOURCE_VERSION}"
    run(
        [
            "./configure",
            f"--prefix={prefix}",
            "--enable-parallel",
            "--enable-shared",
            "--disable-static",
            "--disable-fortran",
            "--disable-cxx",
            "--disable-tests",
            "--disable-tools",
        ],
        cwd=hdf5_src,
        env={**env, "CC": mpicc},
    )
    run(["make", f"-j{os.cpu_count() or 1}"], cwd=hdf5_src, env=env)
    run(["make", "install"], cwd=hdf5_src, env=env)

    print(f"building netCDF-C {NETCDF_C_SOURCE_VERSION} into {prefix}")
    netcdf_archive = work_dir / "netcdf-c.tar.gz"
    download(
        (
            "https://github.com/Unidata/netcdf-c/archive/refs/tags/"
            f"v{NETCDF_C_SOURCE_VERSION}.tar.gz"
        ),
        netcdf_archive,
    )
    extract_tar(netcdf_archive, work_dir)
    netcdf_src = work_dir / f"netcdf-c-{NETCDF_C_SOURCE_VERSION}"
    build_env = dict(env)
    build_env.update(
        {
            "CC": mpicc,
            "CPPFLAGS": f"-I{prefix / 'include'}",
            "LDFLAGS": f"-L{prefix / 'lib'}",
        }
    )
    prepend_library_path(build_env, prefix / "lib")
    run(
        [
            "./configure",
            f"--prefix={prefix}",
            "--disable-static",
            "--enable-shared",
            "--disable-dap",
            "--disable-byterange",
            "--disable-testsets",
            "--disable-utilities",
        ],
        cwd=netcdf_src,
        env=build_env,
    )
    run(["make", f"-j{os.cpu_count() or 1}"], cwd=netcdf_src, env=build_env)
    run(["make", "install"], cwd=netcdf_src, env=build_env)

    meta_header = prefix / "include" / "netcdf_meta.h"
    text = meta_header.read_text()
    if not re.search(r"^#define\s+NC_HAS_PARALLEL4\s+1", text, re.MULTILINE):
        raise SetupError("Source build completed but NC_HAS_PARALLEL4 is not set.")

    result_env = dict(env)
    result_env["NETCDF4_DIR"] = str(prefix)
    result_env["CC"] = mpicc
    prepend_library_path(result_env, prefix / "lib")
    print("source build complete, parallel NetCDF-4 confirmed")
    return result_env


def patch_netcdf4_python_compat(src_dir: Path) -> None:
    compat_header = src_dir / "include" / "netcdf-compat.h"
    if compat_header.is_file():
        text = compat_header.read_text()
        patched = re.sub(
            r"static inline int nc_(def|inq)_var_(bzip2|blosc)\([^{]*\{[^}]*\}\n",
            "",
            text,
        )
        patched = patched.replace(
            "#if NC_VERSION_GE(4, 9, 0)\n#define HAS_NCRCSET 1",
            "#if NC_VERSION_GE(4, 9, 1)\n#define HAS_NCRCSET 1",
        )
        if patched != text:
            compat_header.write_text(patched)

    complex_header = (
        src_dir / "external" / "nc_complex" / "include" / "nc_complex" / "nc_complex.h"
    )
    if complex_header.is_file():
        text = complex_header.read_text()
        patched = text
        for name in ("pfnc_inq_varndims", "pfnc_inq_vardimid"):
            patched = patched.replace(
                f"NC_COMPLEX_EXPORT inline int {name}(",
                f"static inline int {name}(",
            )
        if patched != text:
            complex_header.write_text(patched)


def parallel_netcdf_confirmed(python: Path, env: Dict[str, str]) -> bool:
    result = run(
        [str(python), "-c", PARALLEL_CHECK],
        env=env,
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def build_parallel_io_stack(
    python: Path,
    prefix: Path,
    work_dir: Path,
    base_env: Dict[str, str],
) -> Tuple[Dict[str, str], str, str]:
    module_result = find_module_stack(base_env)
    mpi_module = ""
    netcdf_module = ""
    if module_result is not None:
        build_env, mpi_module, netcdf_module = module_result
    else:
        print("no HPC module stack found; building HDF5/netCDF-C from source")
        build_env = build_source_stack(prefix, work_dir, base_env)

    mpicc = build_env.get("CC") or executable_in_env("mpicc", build_env)
    if not mpicc:
        raise SetupError("Parallel stack selected but mpicc could not be resolved")

    pip_base = [str(python), "-m", "pip"]
    run(
        [
            *pip_base,
            "install",
            "--no-cache-dir",
            "--break-system-packages",
            "--ignore-installed",
            "--upgrade",
            "setuptools>=77",
            "wheel",
            "cython",
        ],
        env=build_env,
    )

    mpi4py_env = dict(build_env)
    mpi4py_env["MPI4PY_BUILD_MPICC"] = mpicc
    run(
        [
            *pip_base,
            "install",
            "--no-cache-dir",
            "--break-system-packages",
            "--no-binary=mpi4py",
            "--no-deps",
            "--force-reinstall",
            "mpi4py",
        ],
        env=mpi4py_env,
    )

    netcdf_src_root = work_dir / "netcdf4-src"
    netcdf_src_root.mkdir(parents=True, exist_ok=True)
    run(
        [
            *pip_base,
            "download",
            "--no-cache-dir",
            "--no-build-isolation",
            "--no-binary=netCDF4",
            "--no-deps",
            "netCDF4",
            "-d",
            str(netcdf_src_root),
        ],
        env=build_env,
    )
    archives = sorted(netcdf_src_root.glob("netcdf4-*.tar.gz"))
    if len(archives) != 1:
        raise SetupError("Expected exactly one downloaded netCDF4 source archive")
    extract_tar(archives[0], netcdf_src_root)
    source_dirs = sorted(
        path for path in netcdf_src_root.glob("netcdf4-*") if path.is_dir()
    )
    if len(source_dirs) != 1:
        raise SetupError("Expected exactly one extracted netCDF4 source directory")
    src_dir = source_dirs[0]
    patch_netcdf4_python_compat(src_dir)
    run(
        [
            *pip_base,
            "install",
            "--no-cache-dir",
            "--break-system-packages",
            "--no-build-isolation",
            "--no-deps",
            "--force-reinstall",
            ".",
        ],
        cwd=src_dir,
        env=build_env,
    )

    if not parallel_netcdf_confirmed(python, build_env):
        raise SetupError("netCDF4 built without parallel4 support")
    run(
        [
            str(python),
            "-c",
            (
                "import netCDF4; "
                "print(f'netCDF4 {netCDF4.__version__}: parallel4 support confirmed')"
            ),
        ],
        env=build_env,
    )
    return build_env, mpi_module, netcdf_module


def write_conda_hooks(
    prefix: Path,
    build_env: Dict[str, str],
    mpi_module: str,
    netcdf_module: str,
) -> None:
    activate_dir = prefix / "etc" / "conda" / "activate.d"
    deactivate_dir = prefix / "etc" / "conda" / "deactivate.d"
    activate_dir.mkdir(parents=True, exist_ok=True)
    deactivate_dir.mkdir(parents=True, exist_ok=True)

    activate_lines = [
        "#!/usr/bin/env bash",
        'export _XGEO_PIO_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"',
        "",
    ]
    if mpi_module:
        activate_lines.append(f"module load {shlex.quote(mpi_module)}")
    if netcdf_module:
        activate_lines.append(f"module load {shlex.quote(netcdf_module)}")

    library_path = build_env.get("LD_LIBRARY_PATH", "")
    if library_path:
        activate_lines.append(
            "export LD_LIBRARY_PATH="
            + f"{shlex.quote(library_path)}"
            + '"${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"'
        )
    else:
        activate_lines.append('export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"')

    activate = activate_dir / "XGEO-parallel-io.sh"
    deactivate = deactivate_dir / "XGEO-parallel-io.sh"
    activate.write_text("\n".join(activate_lines) + "\n")
    deactivate.write_text(
        """#!/usr/bin/env bash
if [[ -n "${_XGEO_PIO_OLD_LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="${_XGEO_PIO_OLD_LD_LIBRARY_PATH}"
else
    unset LD_LIBRARY_PATH
fi
unset _XGEO_PIO_OLD_LD_LIBRARY_PATH
"""
    )
    for path in (activate, deactivate):
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"installed conda activate/deactivate hooks in {prefix}")


def write_virtualenv_hook(prefix: Path, build_env: Dict[str, str]) -> None:
    marker_begin = "# BEGIN XGEO-parallel-io"
    marker_end = "# END XGEO-parallel-io"
    activate_script = prefix / "bin" / "activate"
    if not activate_script.is_file():
        return

    text = activate_script.read_text()
    pattern = re.compile(
        rf"^\s*{re.escape(marker_begin)}.*?^\s*{re.escape(marker_end)}\s*\n?",
        re.MULTILINE | re.DOTALL,
    )
    text = pattern.sub("", text)
    library_path = build_env.get("LD_LIBRARY_PATH", "")
    hook = "\n".join(
        [
            marker_begin,
            (
                "export LD_LIBRARY_PATH="
                f"{shlex.quote(library_path)}"
                '"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"'
            ),
            marker_end,
            "",
        ]
    )
    activate_script.write_text(text.rstrip("\n") + "\n" + hook)
    print(f"appended LD_LIBRARY_PATH export to {activate_script}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Set up XGEO with an MPI-enabled HDF5/netCDF-C/netCDF4 Python stack."
        )
    )
    parser.add_argument(
        "env_name",
        nargs="?",
        default="XGEO",
        help="conda environment name to create when no environment is active",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_dir = Path(__file__).resolve().parent.parent

    with tempfile.TemporaryDirectory(prefix="xgeo-env-") as temporary:
        work_dir = Path(temporary)
        managed_env_name = ""
        is_conda = False
        conda = conda_executable()

        active_conda = os.environ.get("CONDA_DEFAULT_ENV", "")
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        virtualenv = os.environ.get("VIRTUAL_ENV", "")

        if active_conda and active_conda != "base" and conda_prefix:
            prefix = Path(conda_prefix).resolve()
            print(f"using active conda environment: {active_conda}")
            is_conda = True
        elif virtualenv:
            prefix = Path(virtualenv).resolve()
            print(f"using active virtualenv: {prefix.name}")
        else:
            managed_env_name = args.env_name
            print(
                f"no active environment; creating conda environment '{managed_env_name}'"
            )
            if conda is None:
                conda = install_miniconda()

            boot_file = repo_dir / "boot.yaml"
            if not boot_file.is_file():
                raise SetupError(f"Missing conda bootstrap specification: {boot_file}")

            existing = conda_envs(conda)
            if managed_env_name in existing:
                run(
                    [
                        str(conda),
                        "env",
                        "update",
                        "-n",
                        managed_env_name,
                        "-f",
                        str(boot_file),
                    ]
                )
            else:
                run(
                    [
                        str(conda),
                        "env",
                        "create",
                        "-n",
                        managed_env_name,
                        "-f",
                        str(boot_file),
                    ]
                )
            prefix = conda_prefix_for_name(conda, managed_env_name)
            is_conda = True

        python = prefix / "bin" / "python"
        if not python.is_file():
            raise SetupError(f"Target environment Python not found: {python}")

        base_env = dict(os.environ)
        prepend_path(base_env, prefix / "bin")
        build_env, mpi_module, netcdf_module = build_parallel_io_stack(
            python,
            prefix,
            work_dir,
            base_env,
        )

        if is_conda:
            write_conda_hooks(prefix, build_env, mpi_module, netcdf_module)
        else:
            write_virtualenv_hook(prefix, build_env)

        if is_conda and managed_env_name:
            environment_file = repo_dir / "environment.yaml"
            if not environment_file.is_file():
                raise SetupError(
                    f"Missing conda environment specification: {environment_file}"
                )
            if conda is None:
                raise SetupError("Conda executable was unexpectedly unavailable")

            print("applying environment.yaml")
            run(
                [
                    str(conda),
                    "env",
                    "update",
                    "-n",
                    managed_env_name,
                    "-f",
                    str(environment_file),
                ]
            )
            if not parallel_netcdf_confirmed(python, build_env):
                print(
                    "environment.yaml solve replaced the parallel I/O build; restoring it"
                )
                build_env, mpi_module, netcdf_module = build_parallel_io_stack(
                    python,
                    prefix,
                    work_dir,
                    base_env,
                )
                write_conda_hooks(prefix, build_env, mpi_module, netcdf_module)

        print("installing XGEO (editable)")
        run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--break-system-packages",
                "--no-deps",
                "-e",
                str(repo_dir),
            ],
            env=build_env,
        )

        print("done")
        if managed_env_name:
            print(f"Activate with: conda activate {managed_env_name}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SetupError, OSError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
