@echo off

SET VENV_NAME=flask

start /B cmd /k "activate %VENV_NAME% & cd /d %CD%\web_app & python web_app.py"
cmd /k "activate %VENV_NAME% & cd /d %CD%\model_api & python model_api.py"
