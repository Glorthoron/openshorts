#!/usr/bin/env bash
# Se ejecuta una vez, al crear el codespace.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

sudo apt-get update -qq
sudo apt-get install -y -qq zip unzip jq >/dev/null

# El backend corre como appuser (uid 999) y escribe en estos tres directorios,
# que llegan por bind mount y pertenecen al usuario del codespace (uid 1000).
mkdir -p output uploads .os-cache exports
chmod 0777 output uploads .os-cache

chmod +x scripts/os-housekeeping
sudo install -m 0755 scripts/os-housekeeping /usr/local/bin/os-housekeeping

for line in '.os-cache/' 'uploads/' 'exports/'; do
  grep -qxF "$line" .gitignore 2>/dev/null || printf '%s\n' "$line" >> .gitignore
done

if [[ ! -f .env ]]; then
  cp .env.example .env
fi
if [[ -n "${GEMINI_API_KEY:-}" ]] && ! grep -q '^GEMINI_API_KEY=' .env; then
  printf 'GEMINI_API_KEY=%s\n' "$GEMINI_API_KEY" >> .env
  echo "GEMINI_API_KEY copiada del secret del codespace a .env"
fi

if ! docker info >/dev/null 2>&1; then
  echo
  echo "Aviso: el demonio de Docker todavía no responde. El feature docker-in-docker"
  echo "tarda un poco en arrancar. Espera un minuto y ejecuta 'make up'."
fi

cat <<'TXT'

Listo. Primera vez (30-40 min de build, una sola vez):

    make up

El panel queda en el puerto 5175. Lee el README para el flujo semanal.
TXT
