@echo off
IF NOT EXIST env (python3 -m venv env)
call env\Scripts\activate.bat
pip3 install -r scripts/setup/packages.txt
deactivate