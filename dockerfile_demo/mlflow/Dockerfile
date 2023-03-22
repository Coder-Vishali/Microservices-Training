FROM python:3.8.0

# Define the proxies
ENV HTTP_PROXY "<proxy>"
ENV HTTPS_PROXY "<proxy>"

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION "python"

# Set USER
USER root

RUN printf 'Acquire::http::Proxy "<proxy>";\n Acquire::https::Proxy "<proxy>";\n' >> /etc/apt/apt.conf

ENV MLFLOW_HOME /opt/mlflow
ENV MLFLOW_VERSION 2.0.1
ENV SERVER_PORT 5002
ENV SERVER_HOST 0.0.0.0

# Local filestore
# ENV MLFLOW_BACKEND_STORE_URI "file://${MLFLOW_HOME}/mlruns"
# ENV MLFLOW_ARTIFACT_STORE_URI "file://${MLFLOW_HOME}/mlruns"
# ENV MLFLOW_ARTIFACT_STORE_URI "./mlruns"

# Postgresql to log parameters and metrics
ENV MLFLOW_BACKEND_STORE_URI postgresql://<user>:<pass>@<postgres_container_name>:<port_number>/<database_name>
# Jfrog Artifactory to store artifacts
ENV MLFLOW_ARTIFACT_STORE_URI "https://<user>:<pass>@<artifactory_server_name>/<repository_name>/"

RUN pip install --upgrade pip --proxy <proxy>
RUN pip install protobuf==3.20.* --proxy <proxy>
RUN pip install psycopg2-binary --proxy <proxy>
RUN pip install mlflow==${MLFLOW_VERSION} && \
    mkdir -p ${MLFLOW_HOME}/mlflow_server

COPY mlflow_server/run.sh ${MLFLOW_HOME}/mlflow_server/run.sh
RUN chmod +x ${MLFLOW_HOME}/mlflow_server/run.sh

EXPOSE ${SERVER_PORT}/tcp

VOLUME ["${MLFLOW_HOME}/mlflow_server/"]

WORKDIR ${MLFLOW_HOME}

ENTRYPOINT ["./mlflow_server/run.sh"]
