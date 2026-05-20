from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.request import urlopen

# Ensure PyInstaller collects runtime deps (Streamlit + plotting stack).
# We don't use them directly here, but importing makes packaging reliable.
import streamlit  # noqa: F401
import pandas  # noqa: F401
import plotly  # noqa: F401
from streamlit.web import bootstrap


def _resolve_workspace() -> Path:
    raw_candidates = [
        Path.cwd(),
        Path(__file__).resolve().parent,
        Path(sys.executable).resolve().parent,
    ]
    seen: set[str] = set()
    candidates: list[Path] = []
    for base in raw_candidates:
        for p in [base, *base.parents[:4]]:
            key = str(p).lower()
            if key not in seen:
                seen.add(key)
                candidates.append(p)

    for c in candidates:
        if (c / "dashboard.py").exists() and ((c / ".venv311" / "Scripts" / "streamlit.exe").exists() or (c / ".venv" / "Scripts" / "streamlit.exe").exists()):
            return c
    for c in candidates:
        if (c / "dashboard.py").exists():
            return c
    return Path.cwd()


def _find_free_port(preferred: int = 8501) -> int:
    for port in (preferred, 8502, 8503, 8504, 8505):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred


def _streamlit_health_ok(port: int) -> bool:
    # If some other app is using the port, we should not assume it's our dashboard.
    # Streamlit exposes a tiny health endpoint that returns 200 ok.
    try:
        with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=0.6) as r:
            return int(getattr(r, "status", 0) or 0) == 200
    except Exception:
        return False


def main() -> int:
    workspace = _resolve_workspace()
    dashboard = workspace / "dashboard.py"

    no_browser = os.getenv("LAUNCHER_NO_BROWSER", "0") == "1"

    # If the dashboard is already running, just exit.
    # Avoid opening extra browser tabs when the launcher is triggered again
    # (for example during long retrain periods users may click the shortcut twice).
    for p in (8501, 8502, 8503, 8504, 8505):
        if _streamlit_health_ok(p):
            return 0

    port = _find_free_port(8501)
    url = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_SERVER_ADDRESS"] = "127.0.0.1"
    env["STREAMLIT_SERVER_PORT"] = str(port)
    env["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"
    os.environ.update(env)

    def open_browser_when_ready() -> None:
        deadline = time.time() + 180
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    webbrowser.open(url, new=1, autoraise=True)
                    return
            except OSError:
                time.sleep(0.4)

    if not no_browser:
        threading.Thread(target=open_browser_when_ready, daemon=True).start()

    # Prefer launching the project venv streamlit (most stable on this machine).
    streamlit_exe = workspace / ".venv311" / "Scripts" / "streamlit.exe"
    if not streamlit_exe.exists():
        streamlit_exe = workspace / ".venv" / "Scripts" / "streamlit.exe"
    if streamlit_exe.exists() and dashboard.exists():
        logs_dir = workspace / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        out_log = logs_dir / "launcher_streamlit.out.log"
        err_log = logs_dir / "launcher_streamlit.err.log"
        with out_log.open("ab") as out_f, err_log.open("ab") as err_f:
            subprocess.Popen(
                [
                    str(streamlit_exe),
                    "run",
                    str(dashboard),
                    "--server.port",
                    str(port),
                    "--server.headless",
                    "true",
                    "--server.address",
                    "127.0.0.1",
                    "--global.developmentMode",
                    "false",
                ],
                cwd=str(workspace),
                env=env,
                stdout=out_f,
                stderr=err_f,
            )
        # Keep launcher alive briefly to ensure startup, then exit.
        deadline = time.time() + 180
        while time.time() < deadline:
            if _streamlit_health_ok(port):
                return 0
            time.sleep(0.3)
        return 1

    # Fallback path if venv streamlit is unavailable.
    bootstrap.run(
        main_script_path=str(dashboard),
        is_hello=False,
        args=[],
        flag_options={
            "server.headless": True,
            "server.port": port,
            "server.address": "127.0.0.1",
            "global.developmentMode": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
