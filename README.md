# T3M
---
## Set up:

Once installed, you will need to set up a virtual environment. This is where external packages will be stored locally.
Firstly, ensure you have python3 and pip3 installed. Also use pip3 to install virtualenv
Open up a command terminal, navigate to the T3M folder using ```cd <path>``` and type the following:

*On Windows:*
```.\scripts\setup\setup_windows.bat```

*On Mac/Linux:*
```./scripts/setup/setup_unix.sh```

## Adding new packages:
To add a new package, do not use pip directly. Instead, go into ```scripts/setup/packages.txt``` and add the name of the package there. Then rerun the setup.

---
## Run
- To lint and test code, use the following:

*On Windows:*
```.\scripts\setup\setup_windows.bat```

*On Mac/Linux:*
```./scripts/setup/setup_unix.sh```


- To run code, use the following:

*On Windows:*
```.\scripts\run\test_windows.bat```

*On Mac/Linux:*
```./scripts/run/test_unix.sh```

---
# Workflow

## Project Kanban
We will use automated kanban. Please watch:
https://youtu.be/qRdht9CS_No

