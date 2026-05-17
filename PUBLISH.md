# Publishing `graphmemo` to PyPI

If you want developers around the world to be able to type `pip install graphmemo` (or whatever custom name you choose), you need to publish your package to the **Python Package Index (PyPI)**.

Follow these exact steps to publish your library directly from this directory.

## Step 1: Create an Account on PyPI
1. Go to [https://pypi.org/](https://pypi.org/) and register an account.
2. Go to your Account Settings and create an **API Token**. You will need this token later to upload your code.

## Step 2: Install Build Tools
You need the standard Python build tools (`build`) and the upload tool (`twine`). Open your terminal and run:
```bash
pip install --upgrade build twine
```

## Step 3: Check your `setup.py`
Before publishing, make sure your `setup.py` has the correct information.
1. Open `setup.py`.
2. Ensure the `name` is unique on PyPI (e.g., if `graphmemo` is taken, you might need to use `graphmemo_agent` or `agentic_memory`).
3. Ensure the `version` is correct (e.g., `"0.1.0"`). Every time you update the code on PyPI, you **must** increase this version number (e.g., `"0.1.1"`).

## Step 4: Build the Distribution
Run the following command in the root folder (`/home/ravi/Documents/projects/graph-rag`):
```bash
python -m build
```
*This will create a `dist/` folder containing your `.tar.gz` and `.whl` files.*

## Step 5: Upload to PyPI
Use `twine` to securely upload the `dist/` folder to PyPI:
```bash
python -m twine upload dist/*
```
- It will prompt you for a username. Type exactly: `__token__`
- It will prompt you for a password. Paste the **API Token** you generated in Step 1 (it starts with `pypi-...`).

## Step 6: Test Your Install
Wait about 2 minutes for PyPI's servers to cache the new package. Then, go to any other folder on your computer and run:
```bash
pip install graphmemo
```
(Replace `graphmemo` with whatever `name` you put in `setup.py`).

---

### Pro-Tip: Testing on TestPyPI First (Optional)
If you are afraid of messing up the real PyPI repository, you can publish to **TestPyPI** first.
1. Create an account at [https://test.pypi.org/](https://test.pypi.org/)
2. Upload using: `python -m twine upload --repository testpypi dist/*`
3. Install using: `pip install -i https://test.pypi.org/simple/ graphmemo`
