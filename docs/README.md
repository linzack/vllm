# vLLM documents

## Build the docs

<<<<<<< HEAD
```bash
# Install dependencies.
pip install -r requirements-docs.txt

# Build the docs.
make clean
=======
- Make sure in `docs` directory

```bash
cd docs
```

- Install the dependencies:

```bash
pip install -r ../requirements/docs.txt
```

- Clean the previous build (optional but recommended):

```bash
make clean
```

- Generate the HTML documentation:

```bash
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
make html
```

## Open the docs with your browser

<<<<<<< HEAD
=======
- Serve the documentation locally:

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
```bash
python -m http.server -d build/html/
```

<<<<<<< HEAD
Launch your browser and open localhost:8000.
=======
This will start a local server at http://localhost:8000. You can now open your browser and view the documentation.

If port 8000 is already in use, you can specify a different port, for example:

```bash
python -m http.server 3000 -d build/html/
```
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
