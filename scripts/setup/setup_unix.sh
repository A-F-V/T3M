#!/bin/bash
[[ ! -d "env" ]] && python3 -m venv env
source env/bin/activate
pip3 install -r scripts/setup/packages.txt
ipython kernel install --user --name=env
deactivate