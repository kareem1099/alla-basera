#!/bin/bash
set -e

if [ "${RUN_MIGRATIONS:-true}" = "true" ]; then
    echo "Running database migrations..."
    cd /app/models/dbschemas/minirag
    alembic upgrade head
    cd /app
fi

exec "$@"
