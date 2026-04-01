FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY . /opt/GBSeparation
WORKDIR /opt/GBSeparation

RUN pip install --no-cache-dir ".[all]"

ENTRYPOINT ["gbseparation"]
