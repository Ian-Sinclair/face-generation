# Makefile Summary

# This Makefile provides a set of commands to manage a Python project automation, 
# including setting up a virtual environment, installing dependencies, 
# compiling the project, running tests, and building distribution packages.


# Folder name to store virtual environment
VENV=venv
BIN=${VENV}/bin/

NUM_EPOCHS=150

# Define ANSI escape sequences for colors
GREEN := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
RESET := $(shell tput -Txterm sgr0)


#  This target removes build artifacts and directories
#  generated during the build process.
.PHONY: clean_target
clean_target:
	rm -rf build
	rm -rf dist
	rm -rf CNN_Training.egg-info
	rm -rf target
	@echo "$(GREEN)Build Directories Cleaned Successfully$(RESET)"

#  This target removes the virtual environment directory.
.PHONY: clean_venv
clean_venv:
	rm -rf venv
	@echo "$(GREEN)Virtual Environment Cleaned Successfully$(RESET)"

#  This target creates and activates a Python virtual 
#  environment named <venv>.
.PHONY: activate_venv
activate_venv:
	python3 -m venv venv && source venv/bin/activate
	@echo "$(GREEN)Virtual Environment Activated Successfully$(RESET)"

#  This target installs project dependencies listed in 
#  'requirements.txt' into the virtual environment.
.PHONY: install
install: activate_venv requirements.txt
	${BIN}pip3 install -r requirements.txt
	@echo "$(GREEN)Dependencies Installed To ${BIN}$(RESET)"


#  This target installs the project in 'editable' mode, 
#  allowing changes to the source to be immediately reflected 
#  without needing to reinstall the package.
.PHONY: compile
compile: setup.py
	${BIN}pip3 install -e .
	@echo "$(GREEN)Project Source Compiled Successfully$(RESET)"

#  This target installs dependencies, compiles the project, 
#  and runs all unit tests in src/tests directory
.PHONY: test
test: install compile
	${BIN}tox
	@echo "$(GREEN)Completed Tests$(RESET)"



.PHONY: train_mnist
train_mnist: install compile
	${BIN}python3 src/main/scripts/trainingScripts/mnistBasicTrainingCycle.py
	@echo "$(GREEN)Process Completed$(RESET)"

.PHONY: train_anime_faces
train_anime_faces: install compile
	${BIN}python3 src/main/scripts/trainingScripts/animeFacesTrainingCycle.py
	@echo "$(GREEN)Process Completed$(RESET)"

.PHONY: pipeline_train_animeFaces
pipeline_train_animeFaces: install compile
	for ((i=1; i <= ${NUM_EPOCHS}; ++i)) do \
		echo "Initializing Epoch $${i}"; \
		${BIN}python3 src/main/pipelines/animeFacesTrainingPipeline/collect_data.py; \
		${BIN}python3 src/main/pipelines/animeFacesTrainingPipeline/run_epoch_cycle.py; \
	done

#  This target cleans the target directories, installs dependencies, 
#  compiles the project, and builds distribution packages using Python's 
#  build module.
.PHONY: build
build: clean_target install compile setup.py
	${BIN}python3 -m build --outdir target/dist
	@echo "$(GREEN)Project Built Complete$(RESET)"
