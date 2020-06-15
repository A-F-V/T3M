@echo off
WHERE python3
IF %ERRORLEVEL% NEQ 0 doskey python3 = python
IF NOT EXIST env (python3 -m venv env)
call env\Scripts\activate.bat
pip3 install -r scripts/setup/packages.txt
ipython kernel install --user --name=env
deactivate