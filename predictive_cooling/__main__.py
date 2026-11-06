"""`python -m predictive_cooling` — build the study if needed, then open the dashboard."""
import subprocess
import sys

from . import config as C
from . import pipeline


def main():
    if pipeline.load(rebuild_if_stale=False) is None:
        pipeline.build()
    args = sys.argv[1:] or ["--server.port", "8501"]
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(C.ROOT / "app.py"), *args], check=False)


if __name__ == "__main__":
    main()
