"""Desktop entrypoint. Bundled by PyInstaller into a single app per OS.

Runs the same Flask app as the server/Docker deployment, but:
  - stores the database and session key in a proper per-user OS app-data
    directory instead of next to the (often read-only, bundle-temp)
    executable
  - binds only to 127.0.0.1, not the network, since this is meant for
    single-machine use
  - opens a native window (via pywebview) pointed at the local server
    instead of requiring the user to open a browser manually, falling back
    to the system's default browser if no native webview backend is
    available (e.g. missing WebView2 runtime on a bare-bones Windows image)
  - if another instance is already running, just opens a new window against
    it instead of failing to bind the port

Multi-user login/roles are unchanged from the hosted deployment - useful if
more than one person uses the same machine.
"""
import os
import socket
import sys
import threading
import webbrowser

import platformdirs

APP_NAME = 'DLC Surveillance'
DEFAULT_PORT = 8347


def _port_is_free(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) != 0


def main():
    host = '127.0.0.1'
    port = int(os.environ.get('DLC_DESKTOP_PORT', DEFAULT_PORT))

    app_data_dir = platformdirs.user_data_dir(APP_NAME, appauthor=False)
    os.makedirs(app_data_dir, exist_ok=True)

    # Must be set before `import app` - these become the Flask app's
    # instance_path (secret key + sqlite db location, see app.py) and relax
    # the Secure cookie flag, since there's no TLS on localhost.
    os.environ.setdefault('APP_INSTANCE_PATH', app_data_dir)
    os.environ.setdefault('FLASK_ENV', 'development')

    already_running = not _port_is_free(host, port)

    if not already_running:
        from app import app, _reap_idle_cameras
        from init_db import init_database

        init_database()
        threading.Thread(target=_reap_idle_cameras, daemon=True).start()

        def serve():
            from waitress import serve as waitress_serve
            waitress_serve(app, host=host, port=port, threads=8)

        threading.Thread(target=serve, daemon=True).start()

    url = f'http://{host}:{port}'

    try:
        import webview
        webview.create_window(APP_NAME, url, width=1280, height=800, min_size=(900, 600))
        webview.start()
    except Exception as e:
        # No usable native webview backend (e.g. WebView2 runtime missing
        # on Windows) - fall back to the system's default browser instead
        # of failing to start at all.
        print(f'Native window unavailable ({e}), opening in default browser instead.')
        webbrowser.open(url)
        if not already_running:
            # Keep the process (and its background server thread) alive
            # when we can't rely on a window's close event to end it.
            input('Server running - press Enter in this window to quit.\n')


if __name__ == '__main__':
    main()
