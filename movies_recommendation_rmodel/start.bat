@echo off


SET VENV_NAME=flask



cmd /k "activate %VENV_NAME% & python api.py"
