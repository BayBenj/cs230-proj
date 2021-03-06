VENV?=venv_cs230
PROJ?=src/
MAX_COMPLEXITY?=5


.PHONY : install
.PHONY : virtualenv
.PHONY : update
.PHONY : pull
.PHONY : clean-py
.PHONY : clean
.PHONY : fix-style
.PHONY : check-camel-case


install: virtualenv
	pip3 install --upgrade pip
	pip3 install -r requirements.txt


virtualenv:
	pip3 install virtualenv
	python3 -m virtualenv $(VENV) --system-site-packages


update: pull install clean


pull:
	git pull


# remove python bytecode files
clean-py:
	find $(PROJ) -name "*.py[cod]" -exec rm -f {} \;
	find $(PROJ) -name "__pycache__" -type d -exec rm -rf {} \;


# clean out unwanted files
clean: clean-py


# automatically make python files pep 8-compliant
# (see tox.ini for autopep8 constraints)
fix-style:
	autopep8 -r --in-place --aggressive --aggressive $(PROJ)


# finds all strings in project that begin with a lowercase letter, contain only letters and numbers, and contain at least one lowercase letter and at least one uppercase letter.
check-camel-case: clean-py
	grep -rnw $(PROJ) -e "[a-z]\([A-Z0-9]*[a-z][a-z0-9]*[A-Z]\|[a-z0-9]*[A-Z][A-Z0-9]*[a-z]\)[A-Za-z0-9]*"


