COMPOSE := docker compose

.PHONY: help up down build logs status pack clean deep-clean api

help:
	@sed -n 's/^## //p' Makefile | sed 's/:/ —/'

## up: arranca todo (construye la primera vez, 30-40 min)
up:
	$(COMPOSE) up -d --build

## down: para los contenedores
down:
	$(COMPOSE) down

## build: solo construye la imagen
build:
	$(COMPOSE) build

## logs: sigue los logs del backend
logs:
	$(COMPOSE) logs -f backend

## status: qué ocupa el disco
status:
	os-housekeeping status

## pack: junta los clips en ~/exports/clips.zip
pack:
	os-housekeeping pack

## clean: borra clips y temporales (conserva imágenes y modelos)
clean:
	sudo os-housekeeping clean --yes

## deep-clean: además borra imágenes y modelos (obliga a reconstruir)
deep-clean:
	sudo os-housekeeping clean --yes --deep

## api: comprueba que el backend responde
api:
	@curl -sf localhost:8000/health/ready >/dev/null && echo "API OK" || echo "API no responde"
