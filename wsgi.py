"""Production entrypoint. Runs database setup/migration, starts the
background camera-idle reaper, then serves the app with waitress - a
production-grade, pure-Python, cross-platform WSGI server (unlike gunicorn,
it also runs on Windows, so the same entrypoint works whether this is
deployed in Docker, on a Linux/macOS box, or directly on Windows).

Usage:
    python wsgi.py
"""
import os
import threading

from app import app, _reap_idle_cameras
from init_db import init_database

if __name__ == '__main__':
    init_database()
    threading.Thread(target=_reap_idle_cameras, daemon=True).start()

    host = os.environ.get('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_RUN_PORT', 8000))
    threads = int(os.environ.get('WAITRESS_THREADS', 8))

    from waitress import serve
    serve(app, host=host, port=port, threads=threads)
