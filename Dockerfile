# Make dockerfile for fastapi app

FROM python:3.12-slim

WORKDIR /chess

COPY . /chess

RUN pip install -r requirements.txt

ARG API_KEY
ENV API_KEY=$API_KEY

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "45679"]
