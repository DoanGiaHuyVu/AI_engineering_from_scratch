# Recreate the lesson environment

This environment was tested on an Apple Silicon Mac (`arm64`) with CPython
3.14.4. The package versions are recorded in `requirements.txt`.

From this directory, install and select the Python version, create a virtual
environment, and install the recorded packages:

```bash
uv python install 3.14.4
uv venv --python 3.14.4
source .venv/bin/activate
uv pip install -r requirements.txt
```

Check that the active interpreter is inside `.venv` and the packages load:

```bash
python --version
python -c 'import sys, numpy, torch; print(sys.executable); print(numpy.__version__, torch.__version__); print(torch.backends.mps.is_available())'
```

On a Mac with an available MPS backend, the final line prints `True`.
If it prints `False`, the packages may still work on the CPU. The virtual
environment is ignored by Git; keep `requirements.txt` under version control.
