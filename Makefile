# Makefile Summary

# This Makefile provides a set of commands to manage a Python project automation, 
# including setting up a virtual environment, installing dependencies, 
# compiling the project, running tests, and building distribution packages.


# Detect the operating system
ifeq ($(OS),Windows_NT)
    # Windows commands
    VENV=venv
    BIN=${VENV}/Scripts/
    RM=del /q
    MKDIR=mkdir
    PYTHON=python
    PIP=pip
    TOX=tox
else
    # Unix/Linux/MacOS commands
    VENV=venv
    BIN=${VENV}/bin/
    RM=rm -rf
    MKDIR=mkdir -p
    PYTHON=python3
    PIP=pip3
    TOX=${BIN}tox
endif

# Define the number of epochs
NUM_EPOCHS=150

# Define ANSI escape sequences for colors
GREEN := $(shell tput -Txterm setaf 2)
RESET := $(shell tput -Txterm sgr0)



#  This target removes build artifacts and directories
#  generated during the build process.
.PHONY: clean_target
clean_target:
	$(RM) build dist CNN_Training.egg-info target
	@echo "$(GREEN)Build Directories Cleaned Successfully$(RESET)"

#  This target removes the virtual environment directory.
.PHONY: clean_venv
clean_venv:
	$(RM) $(VENV)
	@echo "$(GREEN)Virtual Environment Cleaned Successfully$(RESET)"

#  This target creates and activates a Python virtual 
#  environment named <venv>.
.PHONY: activate_venv
activate_venv:
	$(PYTHON) -m venv $(VENV) && . $(BIN)activate
	@echo "$(GREEN)Virtual Environment Activated Successfully$(RESET)"

#  This target installs project dependencies listed in 
#  'requirements.txt' into the virtual environment.
.PHONY: install
install: activate_venv requirements.txt requirements-mac-metal.txt
ifeq ($(shell uname),Darwin)
	$(BIN)$(PIP) install -r requirements-mac-metal.txt
else
	$(BIN)$(PIP) install -r requirements.txt
endif
	@echo "$(GREEN)Dependencies Installed Successfully$(RESET)"


#  This target installs the project in 'editable' mode, 
#  allowing changes to the source to be immediately reflected 
#  without needing to reinstall the package.
.PHONY: compile
compile: setup.py
	$(BIN)$(PIP) install -e .
	@echo "$(GREEN)Project Source Compiled Successfully$(RESET)"

#  This target installs dependencies, compiles the project, 
#  and runs all unit tests in src/tests directory
.PHONY: test
test: install compile
	$(TOX)
	@echo "$(GREEN)Completed Tests$(RESET)"

#  Description: Target to train the MNIST dataset.
.PHONY: train_mnist
train_mnist: install compile
	$(BIN)$(PYTHON) src/main/scripts/trainingScripts/mnistBasicTrainingCycle.py
	@echo "$(GREEN)Process Completed$(RESET)"

#  Description: Target to train the anime faces dataset.
.PHONY: train_anime_faces
train_anime_faces: install compile
	$(BIN)$(PYTHON) src/main/scripts/trainingScripts/animeFacesTrainingCycle.py
	@echo "$(GREEN)Process Completed$(RESET)"

#  Description: Target to train the celebrity faces dataset.
.PHONY: train_celeb_faces
train_celeb_faces: install compile
	$(BIN)$(PYTHON) src/main/scripts/trainingScripts/celebFacesTrainingCycle.py
	@echo "$(GREEN)Process Completed$(RESET)"

#  Description: Target to run the training pipeline for anime faces dataset.
.PHONY: pipeline_train_animeFaces
pipeline_train_animeFaces: install compile
	for ((i=1; i <= ${NUM_EPOCHS}; ++i)) do \
		echo "Initializing Epoch $${i}"; \
		$(BIN)$(PYTHON) src/main/pipelines/animeFacesTrainingPipeline/collect_data.py; \
		$(BIN)$(PYTHON) src/main/pipelines/animeFacesTrainingPipeline/run_epoch_cycle.py; \
	done

#  Description: Target to run the training pipeline for celebrity faces dataset.
.PHONY: pipeline_train_celebFaces
pipeline_train_celebFaces: install compile
	for ((i=1; i <= ${NUM_EPOCHS}; ++i)) do \
		echo "Initializing Epoch $${i}"; \
		$(BIN)$(PYTHON) src/main/pipelines/celebFacesTrainingPipeline/collect_data.py; \
		$(BIN)$(PYTHON) src/main/pipelines/celebFacesTrainingPipeline/run_epoch_cycle.py; \
	done

#  This target cleans the target directories, installs dependencies, 
#  compiles the project, and builds distribution packages using Python's 
#  build module.
.PHONY: build
build: clean_target install compile setup.py
	$(BIN)$(PYTHON) -m build --outdir target/dist
	@echo "$(GREEN)Project Built Complete$(RESET)"
