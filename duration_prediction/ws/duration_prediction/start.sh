#!/usr/bin/env bash

#SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
PROJECT_DIR="/app/duration_prediction"

#echo $PROJECT_DIR 

cd "${PROJECT_DIR}/web_app/"
python3 web_app.py &

# ml web service
cd "${PROJECT_DIR}/model_api/"
python3 model_api.py