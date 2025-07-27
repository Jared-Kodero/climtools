import subprocess


def which(cmd):
    try:
        path = (
            subprocess.check_output(["which", cmd], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )

        return True if path else False
    except subprocess.CalledProcessError:
        return None


def eval_pkg_cdo():
    STATE = which("cdo")

    if not STATE:

        print(
            "Please run `conda install -c conda-forge cdo -y` to install CDO with conda",
            "or run `pip install cdo` to install CDO with pip",
        )

    return STATE
