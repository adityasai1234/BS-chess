# Make dockerfile for fastapi app

FROM python:3.12-slim

WORKDIR /chess

COPY . /chess

RUN pip install -r requirements.txt

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "45679"]
