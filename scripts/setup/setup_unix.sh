#!/bin/bash
shopt -s expand_aliases
command -v python >/dev/null 2>&1 || { alias python=python3; }
command -v pip >/dev/null 2>&1 || { alias pip=pip3; }
[[ ! -d "env" ]] && python -m venv env
source env/bin/activate
pip install -r requirements.txt
ipython kernel install --user --name=env
deactivate