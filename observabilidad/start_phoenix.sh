#!/usr/bin/env bash
set -e

CONTAINER_NAME="phoenix"
IMAGE="arizephoenix/phoenix:latest"
PORT="6006"

# Si ya existe corriendo, no hace nada
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Phoenix ya está corriendo en http://localhost:${PORT}"
  exit 0
fi

# Si existe detenido, lo borra
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

docker run -d --rm \
  --name "${CONTAINER_NAME}" \
  -p "${PORT}:${PORT}" \
  "${IMAGE}"

echo "Phoenix levantado en http://localhost:${PORT}"
