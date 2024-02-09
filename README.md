# face-generation


## Operational Pre-requisites

Build automation is run with a `Makefile` and so you must be able to run Makefiles

If you have linux, disregard the next prerequite steps.

### Install `make`  

For Mac OS  

```console
brew install make
```

For Windows  

```console
choco install make
```

## Initialize Project

cd into the root repo directory and run `make install` to install packages and compile project packages.  
Note: Project dependencies will be stored in a virtual environment `venv` tailered to the repository.

## Other Makefile and Project Management Commands

Makefile Usage:
To clean build artifacts: `make clean_target`  
To clean the virtual environment: `make clean_venv`  
To set up the virtual environment and install dependencies: `make install`  
To compile the project: `make compile`  
To run tests: `make test`  
To build distribution packages: `make build`  
