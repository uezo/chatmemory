FROM python:3.11-bullseye

RUN apt-get update && apt-get install -y \
    nano git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY run.py /app/

RUN pip install --upgrade pip \
  && pip install python-dotenv chatmemory -U

EXPOSE 8000
