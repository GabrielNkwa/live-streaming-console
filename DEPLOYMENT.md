# Deploying DLC Surveillance

This app is a multi-user web service (login, roles, per-user cameras), so it's
meant to run **once, continuously, on a machine that can reach your
cameras** - everyone else just opens a browser. Since most RTSP/webcam
sources live on a home or office LAN, that machine should generally be
something on that same network (a mini-PC, NAS, Raspberry Pi 4/5, or an old
desktop left running) rather than a random cloud VPS that can't see your
cameras at all.

## Quick start (Docker, any host OS)

Requires Docker + Docker Compose on the host (Windows, macOS, or Linux -
Docker abstracts the OS away here).

```bash
cp .env.example .env
# edit .env: set APP_ADMIN_PASS at minimum
docker compose up -d --build
```

The app is now listening on port 8000. First startup creates the database
(`instance/dlc.db`) and the admin account from `.env` automatically.

The `instance/` directory (bind-mounted into the container) holds your
database and an auto-generated session secret key - **back it up**, and
don't delete it between deploys or you'll lose all users/cameras and log
everyone out.

**Local webcams (`webcam:N`) vs RTSP/IP cameras:** RTSP cameras are reached
over the network and work regardless of host OS. A USB webcam plugged into
the Docker host only works if that host is Linux with the `devices:` line in
`docker-compose.yml` uncommented - Docker Desktop on Windows/macOS can't
pass USB video devices through to a container. If you specifically want to
view a webcam attached to a Windows or Mac machine, run the app directly on
that machine instead of in Docker (see "Running without Docker" below).

## How to make it reachable

Pick one:

**Tailscale (recommended)** - no public exposure at all. Install Tailscale
on the host and on every device that should have access; they join your
private mesh network and reach the app via its Tailscale IP/hostname on port
8000. Given this app serves live camera footage and handles logins, avoiding
any public internet exposure is the safest default. Nothing in this repo
needs to change for this option.

**Reverse proxy + real domain** - if you want it reachable from anywhere
without Tailscale installed on every device. Use the commented-out `caddy`
service in `docker-compose.yml` and `Caddyfile.example` (copy to
`Caddyfile`, set your domain, then `docker compose --profile proxy up -d`).
Caddy handles HTTPS certificates automatically. If you do this, also set
`USE_PROXY_FIX=1` in `.env` so the app trusts the proxy's forwarded headers.

**Plain LAN access** - just open the host's port directly to your home
network (`ports: "8000:8000"` already does this). Fine for a trusted home
network; the app enforces HTTPS-oriented cookie flags once `FLASK_ENV` is
`production`, so browsers accessing it over plain `http://` on your LAN may
warn about insecure cookies in some configurations - Tailscale or Caddy
avoid that by giving you real TLS.

## Updating

```bash
git pull
docker compose up -d --build
```

Your data survives since `instance/` is a bind mount, not part of the image.

## Running without Docker

Works the same way on Windows, macOS, or Linux directly:

```bash
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env  # then export/set those variables in your shell, or use a tool like python-dotenv
python wsgi.py
```

## Sizing

torch + ultralytics + OpenCV are heavy. The web app and RTSP/webcam
streaming (AI detection off) run fine on something as small as a Raspberry
Pi 4. Running AI-mode (YOLO) detection is CPU-bound and will be slow
(low single-digit FPS) on Pi-class hardware; a small x86 box with 4+ cores
and 8GB+ RAM handles a few concurrent AI streams comfortably. There's no GPU
acceleration wired up by default - all inference is CPU-only.
