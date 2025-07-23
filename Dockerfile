# app/Dockerfile

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Google Chrome
RUN wget -O /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    apt-get update && \
    apt-get install -y /tmp/chrome.deb && \
    rm /tmp/chrome.deb

COPY . .

RUN pip3 install -r requirements.txt
RUN pip3 install -U kaleido

EXPOSE 8080

HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health

ENTRYPOINT ["streamlit", "run", "maker.py", "--server.port=8080", "--server.address=0.0.0.0"]