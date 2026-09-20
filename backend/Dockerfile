# Build from the PROJECT ROOT, not this directory:
#   docker build -f backend/Dockerfile -t financial-analyst-backend .
# (docker-compose.yml already does this correctly)

FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY agent/ agent/
COPY stores/ stores/
COPY backend/main.py backend/main.py
COPY data/ data/

EXPOSE 8000

# Shell form (not exec/JSON form) deliberately — needed for ${PORT:-8000}
# env substitution, since most container hosts (Render included) inject
# PORT at runtime rather than using a fixed one. The leading `exec` still
# makes uvicorn take over the shell's PID directly (rather than running as
# its child), so SIGTERM on container stop reaches uvicorn properly for a
# clean shutdown instead of being swallowed by the shell.
CMD exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
