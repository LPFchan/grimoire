# DEC-20260525-003: Persistent State — Bind Mounts, Hourly Backups, and Operational Safeguards

Opened: 2026-05-26 03:59:26 KST
Recorded by agent: opencode

## Metadata

- Status: proposed
- Deciders: operator
- Related ids: none

## Decision

Move storage — place the grimoire state directory on the host with a bind mount. Add hourly hardlink‑based snapshots to a separate physical disk, retaining the last 24 copies. Codify rules preventing ad‑hoc image mutation. Simplify the systemd unit.

## Context

On 2026-05-25 a series of `docker commit` operations silently mutated the grimoire image's entrypoint and required multiple restarts. During those restarts the Docker-managed named volume was torn down and re‑created.

Three root causes made this destructive:

1. **Docker-managed volume** (`grimoire_state` in `docker-compose.yml`): invisible from the host, destroyed by any compose-project-name change, a stray `docker volume prune`, or a compose‑down‑v. No way to inspect it without `docker exec`.
2. **No backups**: no automated cron, timer, or snapshot was pointed at the state directory.
3. **`docker commit`**: produces irreproducible images, silently mutates entrypoint / cmd, and leaves no audit trail. In this case the entrypoint was changed from `python -m grimoire.entrypoint` to `/opt/grimoire-venv/bin/pip`.

## Design

### 1. Host bind mount for `/var/lib/grimoire`

Replace the Docker volume with a plain host directory:

```yaml
# docker-compose.yml — before
volumes:
  - grimoire_state:/var/lib/grimoire
…
volumes:
  grimoire_state:

# after
volumes:
  - ./state:/var/lib/grimoire
# Docker handles restarts; systemd handles initial start + stop
restart: unless-stopped
```

- Host directory: `~/grimoire/state/`.
- Before applying, copy any surviving data from the current Docker volume:
  ```bash
  mkdir -p ~/grimoire/state
  cp -a /var/lib/docker/volumes/grimoire_grimoire_state/_data/* ~/grimoire/state/
  ```
- Docker will create the directory itself on first `up`; pre‑creating it ensures the correct ownership. All sqlite files are owned by `root:root` inside the container, so set `chown` that matches the container's runtime user if needed (the grimoire container runs as `root`, so root‑owned files are correct).

The bind‑mounted directory is visible on the host and survives any Docker operation — `compose down`, `volume prune`, project rename, or even Docker uninstallation.

### 2. Hourly hardlink‑based snapshot rotation

A systemd timer + oneshot service creates immutable hourly snapshots on `/mnt/MX500`:

**`/etc/systemd/system/grimoire-backup.service`**:
```
[Unit]
Description=Grimoire state hourly snapshot
After=local-fs.target
RequiresMountsFor=/mnt/MX500

[Service]
Type=oneshot
User=root
ExecStartPre=/bin/sh -c 'mountpoint -q /mnt/MX500 || exit 1'
ExecStart=/bin/sh -c '\
  mkdir -p /mnt/MX500/backups/grimoire && \
  TS=$$(date +%%Y%%m%%d-%%H) && \
  SRC=/home/yeowool/grimoire/state && \
  DST=/mnt/MX500/backups/grimoire/state-$$TS && \
  PREV=$$(ls -d /mnt/MX500/backups/grimoire/state-* 2>/dev/null | tail -1); \
  if [ -n "$$PREV" ] && [ -d "$$PREV" ]; then \
    cp -al "$$PREV" "$$DST" && \
    rsync -a --delete "$$SRC/" "$$DST/"; \
  else \
    rsync -a "$$SRC/" "$$DST/"; \
  fi && \
  cd /mnt/MX500/backups/grimoire && \
  ls -d state-* 2>/dev/null | sort | head -n -24 | xargs -r rm -rf'
```

**`/etc/systemd/system/grimoire-backup.timer`**:
```
[Unit]
Description=Hourly grimoire state snapshot

[Timer]
OnCalendar=hourly
RandomizedDelaySec=120
Persistent=true

[Install]
WantedBy=timers.target
```

Key choices:
- **Hardlink snapshots** (`cp -al`): each snapshot is a complete tree, but unchanged files share inodes — space‑efficient and each copy is independently restorable.
- **No `--delete` on the first sync**: the previous snapshot is hardlinked, then only changed files are overwritten by `rsync`. Corrupted or deleted files in the source do not propagate to past snapshots.
- **`RequiresMountsFor=/mnt/MX500`** + **`mountpoint -q` check**: prevents silent writes to the root filesystem when the MX500 is unmounted.
- **`User=root`**: the state files are root‑owned (Docker container runs as root).
- **Retention**: 24 hourly snapshots (1 day of history). The purge step removes all directories older than the 24‑most‑recent. Purge only runs if the snapshot step succeeded (`&&` chain).
- **`Persistent=true`**: catches up on missed backup ticks after the system was down.

### 3. Operational rule — no `docker commit`

The following rule is added to `AGENTS.md` under a new **Operational Rules** heading:

> **Never use `docker commit` on the grimoire image.**
>
> Image changes go through `Dockerfile` → `docker compose build`. For live development code changes, use the existing `DEV_SRC_BIND` mount. For dependency changes, edit `pyproject.toml` and rebuild.
>
> `docker commit` silently mutates image configuration (entrypoint, cmd, env, exposed ports) without audit trail and produces irreproducible images.

### 4. Systemd unit simplification

The current `ExecStartPre=docker compose down` / `ExecStart=docker compose up` pattern is replaced with a single `up --wait`:

```
[Service]
Type=oneshot
RemainAfterExit=yes
TimeoutStartSec=900
TimeoutStopSec=120
User=yeowool
WorkingDirectory=/home/yeowool/grimoire
ExecStart=/usr/bin/docker compose up -d --wait
ExecStop=/usr/bin/docker compose stop
```

- `up -d --wait`: starts container in the background, blocks until healthy. Combined with `Type=oneshot` + `RemainAfterExit=yes`, systemd considers the service active after the healthcheck passes.
- **Docker handles restarts**: the `docker-compose.yml` grimoire service adds `restart: unless-stopped`. If the container crashes (OOM, Python fault, GPU error), Docker restarts it — not systemd. This avoids the `Type=simple` foregound-monitoring complexity entirely.
- `docker compose stop` stops the container without destroying it, networks, or the bind mount.

## Migration Steps

1. `sudo systemctl stop grimoire` (stop container before copying to avoid sqlite lock issues)
2. `mkdir -p ~/grimoire/state`
3. Copy live volume data: `cp -a /var/lib/docker/volumes/grimoire_grimoire_state/_data/* ~/grimoire/state/`
4. Edit `docker-compose.yml` — replace named volume with `./state:/var/lib/grimoire`; add `restart: unless-stopped` to the grimoire service so Docker handles container restarts independently of systemd
5. Edit `/etc/systemd/system/grimoire.service` — simplify ExecStart/ExecStop
6. `sudo systemctl daemon-reload`
7. Create `/etc/systemd/system/grimoire-backup.service` and `grimoire-backup.timer`
8. `sudo systemctl enable --now grimoire-backup.timer`
9. `sudo systemctl restart grimoire`
10. Edit `AGENTS.md` — add operational rule
11. Verify: `ls ~/grimoire/state/` shows database files, `ls /mnt/MX500/backups/grimoire/` shows snapshots appearing

## Files Changed

| File | Action |
|------|--------|
| `docker-compose.yml` | Replace named volume with host bind mount `./state:/var/lib/grimoire`; add `restart: unless-stopped` for Docker-native crash recovery |
| `/etc/systemd/system/grimoire.service` | Simplify ExecStart/ExecStop |
| `/etc/systemd/system/grimoire-backup.service` | Create |
| `/etc/systemd/system/grimoire-backup.timer` | Create |
| `AGENTS.md` | Add operational rule prohibiting `docker commit` |

## Consequences

- State data is on the host filesystem — visible, inspectable, and backup‑able with standard tools
- 24 hourly hardlink‑based snapshots on `/mnt/MX500` — each copy is independently restorable
- Backup timer is self‑healing (`Persistent=true`) and mount‑aware (`RequiresMountsFor` + `mountpoint -q`)
- `docker commit` is documented as forbidden, not merely aspirational — agents are instructed, human operators are warned
- Systemd unit is simpler with fewer moving parts
- Migration preserves any surviving volume data before switching to the bind mount
