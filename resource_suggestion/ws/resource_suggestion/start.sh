#!/usr/bin/env bash

PROJECT_DIR="/app/resource_suggestion"

cd "${PROJECT_DIR}/web_app/"
python3 web_app.py &

# ml web service
cd "${PROJECT_DIR}/model_api/"
python3 model_api.py