### Getting started

To ensure reproducibility, we are using [poetry](https://python-poetry.org/) to manage dependencies. 
To install poetry, follow the instructions [here](https://python-poetry.org/docs/#installation). 
Once poetry is installed, run the following command to install the dependencies:

```bash
poetry install
```

To test the installation, run the following command:

```bash
poetry run python main.py
```

You should see a summary of a sample model.

Poetry automatically creates a virtual environment for the project.
When developing in an IDE, make sure to select the interpreter in the virtual environment.
To find the path to the virtual environment, run the following command:

```bash
poetry env info -p
```

Further, you need to ensure that the `neat/python` folder in contained in the `PYTHONPATH` environment variable.
