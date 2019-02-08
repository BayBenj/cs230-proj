VENV?=venv_cs230_$(SYS_TYPE)
PIP?=$(VENV)/bin/pip
PYTH?=$(VENV)/bin/python
PROJ?=src/
MAX_COMPLEXITY?=5


.PHONY : virtualenv
.PHONY : update
.PHONY : pull
.PHONY : clean-py
.PHONY : clean
.PHONY : fix-style
.PHONY : check-style
.PHONY : check-camel-case
.PHONY : checks


virtualenv:
    python -m virtualenv $(VENV) --system-site-packages


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


# run code style checks
check-style:
    -$(PYTH) -m flake8 --max-complexity $(MAX_COMPLEXITY) $(PROJ)
    -pylint $(PROJ)


# finds all strings in project that begin with a lowercase letter, contain only letters and numbers, and contain at least one lowercase letter and at least one uppercase letter.
check-camel-case: clean-py
    grep -rnw $(PROJ) -e "[a-z]\([A-Z0-9]*[a-z][a-z0-9]*[A-Z]\|[a-z0-9]*[A-Z][A-Z0-9]*[a-z]\)[A-Za-z0-9]*"


# run all checks
checks: check-style check-camel-case

