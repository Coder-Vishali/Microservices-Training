#!/bin/sh

mlflow server \
    --backend-store-uri $MLFLOW_BACKEND_STORE_URI \
    --default-artifact-root $MLFLOW_ARTIFACT_STORE_URI \
    --host $SERVER_HOST \
    --port $SERVER_PORT
