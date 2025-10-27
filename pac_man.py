import subprocess


def which(cmd: str) -> bool | None:
    try:
        path = (
            subprocess.check_output(["which", cmd], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )

        return True if path else False
    except subprocess.CalledProcessError:
        return None
