# Build from the PROJECT ROOT:
#   docker build -f frontend/Dockerfile -t financial-analyst-frontend .

FROM python:3.12-slim

WORKDIR /app

COPY frontend/requirements.txt frontend/requirements.txt
RUN pip install --no-cache-dir -r frontend/requirements.txt

COPY frontend/app.py frontend/app.py

EXPOSE 8501

CMD exec streamlit run frontend/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0
