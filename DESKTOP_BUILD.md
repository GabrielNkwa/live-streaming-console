# Building the Desktop App

This packages the same app as a standalone desktop application (Windows,
macOS, or Linux) using PyInstaller + pywebview: double-click to run, no
Docker, no separate browser tab. It keeps the same multi-user login/roles as
the hosted deployment - useful if more than one person uses the machine.

Each person's install is independent: its own local database, own admin
account, own cameras. This is a different shape from the Docker/hosted
deployment (one shared server, many remote users) - see `DEPLOYMENT.md` if
that's what you actually want instead.

## Quick build (same OS you're building for)

PyInstaller does **not** cross-compile - you must build on the OS you're
targeting (build on Windows for a Windows .exe, on macOS for a macOS app,
etc.). To build all three, either use three machines/VMs, or push a
`v*.*.*` git tag and let `.github/workflows/build-desktop.yml` build all
three in CI and upload them as artifacts.

```bash
python -m venv build-venv
source build-venv/bin/activate  # build-venv\Scripts\activate on Windows
pip install -r requirements-desktop.txt
pyinstaller dlc-surveillance.spec --noconfirm
```

Output lands in `dist/DLC Surveillance/` - a folder, not a single file (see
"Why onedir, not onefile" below). Zip that folder up, or wrap it with a real
installer (Inno Setup on Windows, a signed .app in a .dmg on macOS, an
AppImage or .deb on Linux) for actual distribution - none of that installer
tooling is set up yet, this gets you the working, runnable app folder.

### Platform-specific notes

- **Windows**: needs the Microsoft Edge WebView2 Runtime for the native
  window. It ships by default on Windows 10 (1803+)/11; if it's ever
  missing, the app falls back to opening your default browser instead of
  failing to start.
- **macOS**: uses the built-in WebKit view (no extra runtime needed). You'll
  likely need to right-click → Open the first time to get past Gatekeeper,
  since the build isn't code-signed/notarized.
- **Linux**: pywebview needs a GTK+WebKit2 (or Qt+WebEngine) backend
  installed at the *system* level - pip can't provide this. On
  Debian/Ubuntu: `sudo apt-get install python3-gi gir1.2-gtk-3.0
  gir1.2-webkit2-4.1`. If that backend isn't available, the app falls back
  to your default browser rather than crashing.

## What's different from the hosted/Docker deployment

- Database and session key live in your OS's per-user app-data directory
  (`%LOCALAPPDATA%\DLC Surveillance` on Windows,
  `~/Library/Application Support/DLC Surveillance` on macOS,
  `~/.local/share/DLC Surveillance` on Linux), not next to the executable -
  so they survive reinstalls/updates and don't need admin rights to write.
- Binds only to `127.0.0.1` - not reachable from other devices on your
  network, unlike the Docker deployment.
- Opening the app a second time (while it's already running) just opens
  another window against the already-running server instead of erroring.
- No AI/GPU acceleration difference from the server build - still CPU-only
  YOLO inference either way.

## Why onedir, not onefile

PyInstaller's "onefile" mode re-extracts the entire bundle to a temp
directory on *every single launch*. That's fine for a small script; for a
multi-GB bundle (torch + opencv + ultralytics), it means a genuinely slow
startup every time. `dlc-surveillance.spec` builds "onedir" instead -
extraction happens once at build time, and launches after that are fast.
The tradeoff is you're distributing a folder instead of one .exe; a real
installer (see above) hides that from end users anyway.

## Rebuilding after code changes

Just rerun `pyinstaller dlc-surveillance.spec --noconfirm` - it picks up
changes to `app.py`, `desktop.py`, `templates/`, `static/`, and
`yolo11n.pt` automatically since they're all referenced by the spec file.

## Debugging a build

`dlc-surveillance.spec` has `console=True`, which keeps a console window
open showing stdout/stderr/tracebacks - useful while getting a build
working. Flip it to `console=False` for a clean release build once you've
confirmed it works, so end users don't see a console window alongside the
app window.

If `collect_all()` in the spec ever misses something PyInstaller can't
auto-detect (shows up as an ImportError only when running the *built* exe,
not when running `python desktop.py` directly), add the missing module name
to the `hiddenimports` list in `dlc-surveillance.spec`.
