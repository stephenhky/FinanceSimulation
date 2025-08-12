# finsim Documentation

This directory contains the documentation for the finsim project.

## Building the Documentation

To build the documentation, you can use the provided script:

```bash
python make_docs.py
```

This will generate HTML documentation in the `_build/html` directory.

Alternatively, you can use the standard Sphinx make commands:

```bash
make html
```

## Dependencies

The documentation requires the following dependencies:

- sphinx>=4.0.0
- sphinx-rtd-theme>=1.0.0

These can be installed with:

```bash
pip install -r requirements.txt
```

Or if you're installing finsim with documentation support:

```bash
pip install finsim[docs]
```

## Structure

- `conf.py`: Sphinx configuration file
- `index.rst`: Main documentation page
- `modules.rst`: API reference documentation
- `make_docs.py`: Script to automatically generate documentation
- `requirements.txt`: Documentation dependencies
- `_build/`: Generated documentation output
- `_static/`: Static files for the documentation
- `_templates/`: Templates for the documentation