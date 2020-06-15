@echo off
WHERE python
IF %ERRORLEVEL% NEQ 0 doskey python = python3
WHERE pip
IF %ERRORLEVEL% NEQ 0 doskey pip = pip3
IF NOT EXIST env (python -m venv env)
call env\Scripts\activate.bat
pip install -r scripts/setup/packages.txt
ipython kernel install --user --name=env
deactivate