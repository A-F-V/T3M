#!/bin/bash
command -v python >/dev/null 2>&1 || { alias python=python3; }
command -v pip >/dev/null 2>&1 || { alias pip=pip3; }
[[ ! -d "env" ]] && python -m venv env
source env/bin/activate
pip install -r scripts/setup/packages.txt
ipython kernel install --user --name=env
deactivate