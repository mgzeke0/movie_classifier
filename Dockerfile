FROM python:3.6-slim
ARG PORT=8081
RUN apt-get update && apt-get install --yes gcc
#COPY rest_service/ /service/rest_service/
#COPY model/ /service/model/
#COPY serve.py config.py requirements.txt /service/
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . /service
WORKDIR /service
ENV SERVICE_PORT=$PORT
EXPOSE $SERVICE_PORT
ENTRYPOINT uvicorn rest_service.server_rest:app --host "0.0.0.0" --port $SERVICE_PORT
