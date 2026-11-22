#!/usr/bin/env bash
# Se ejecuta en cada arranque del codespace.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

mkdir -p output uploads .os-cache
mkdir -p "$HOME/exports"
chmod 0777 output uploads .os-cache 2>/dev/null || true

# Restos de la sesión anterior: si ya los descargaste, sobran.
os-housekeeping prune-stale --hours 6 --quiet || true

# Limpieza pasiva mientras la sesión está abierta (cada 30 min, solo si no hay
# ningún clip en marcha).
os-housekeeping watchdog --daemon || true

os-housekeeping status --short || true
