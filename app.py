import nbformat

path = "FairVision.ipynb"

nb = nbformat.read(path, as_version=4)

if "widgets" in nb["metadata"]:
    del nb["metadata"]["widgets"]

nbformat.write(nb, path)

print("Fixed notebook")