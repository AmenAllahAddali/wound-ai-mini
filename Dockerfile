# Image de l'API Wound-AI Mini (FastAPI + U-Net, CPU)
FROM python:3.12-slim

# Bibliotheques systeme necessaires a OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependances Python (mises en cache tant que requirements.txt ne change pas)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code et modele entraine
COPY src/ src/
COPY api/ api/
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
