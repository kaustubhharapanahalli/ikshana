# Contribution Guide

## Environment setup

This project uses tools like black, isort, pylint, pydocstyle, mypy, pre-commit to ensure maintenance of code quality.

### Pre-requisites

1. Install [Python](https://www.python.org/downloads) on your local machine.
2. If you are using Windows or Mac for development, install [Docker Desktop](https://www.docker.com/products/docker-desktop). If you are using Ubuntu, install [Docker](https://docs.docker.com/engine/install/ubuntu/).

### Development

1. Setup Poetry by following the poetry installation guide: [Poetry installation documentation](https://python-poetry.org/docs/)

2. Install all the required packages:

   ```bash
   poetry install
   ```

   This will create a virtual environment and all the packages will be installed in the virtual environment folder. To run any command related to the project, you need to prepend the commands with `poetry run ...`.

3. Install Git pre-commit hooks to enable code quality checks when making a Git commit:

   ```bash
   poetry run pre-commit install
   ```
