@echo off
call env\Scripts\activate.bat
pytest
deactivate