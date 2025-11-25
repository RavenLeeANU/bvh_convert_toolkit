
### Install Environment

pls refer to the `https://pypi.org/project/bpy/4.2.0/#files` to see the bpy version



```bash

conda create -n bvh_converter python=3.11

pip install -r requirements.txt

```

### Pack EXE

```bash
pyinstaller --onefile --collect-submodules "bpy" --collect-all "bpy" app.py
```