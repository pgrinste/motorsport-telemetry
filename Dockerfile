# Suzuka Telemetry — lightweight backend + static frontend.
# GPU work (Monte Carlo) is precomputed; runtime only needs CPU Python.
FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" websockets pandas numpy

COPY server/app.py ./server/
COPY frontend/ ./frontend/
COPY build/bundle.json ./build/
COPY build/czml/ ./build/czml/
COPY outputs/telemetry_driver.csv ./outputs/

EXPOSE 8321
WORKDIR /app/server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8321"]
