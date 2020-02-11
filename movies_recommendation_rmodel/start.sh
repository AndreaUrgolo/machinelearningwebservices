#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
PROJECT_DIR=$SCRIPTPATH

cd "${PROJECT_DIR}/web_app/"
python web_app.py &

# ml web service
cd "${PROJECT_DIR}/model_api/"
python model_api.py
