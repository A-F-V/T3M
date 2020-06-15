#!/bin/bash
command -v python >/dev/null 2>&1 || { alias python=python3; }
[[ ! -d "env" ]] && python3 -m venv env
source env/bin/activate
pip3 install -r scripts/setup/packages.txt
ipython kernel install --user --name=env
deactivate