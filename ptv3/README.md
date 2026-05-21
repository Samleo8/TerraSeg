# PTv3 (Point Transformer v3)


This directory packages the Point Transformer v3 (PTv3) backbone [[paper](https://arxiv.org/abs/2312.10035), [code](https://github.com/Pointcept/PointTransformerV3)] as a workspace member so it can be imported by TerraSeg with a pinned, reproducible version.


## Pinned upstream commit


We pin PTv3 to upstream commit **`3229e9b7de1770c8ad17c316f8e349982de509f8`** of [`Pointcept/PointTransformerV3`](https://github.com/Pointcept/PointTransformerV3).


At this commit the upstream repository is not a pip-installable Python package, it ships only `model.py` and a `serialization/` folder, with the integration guidance "*copy these files into your project*". We therefore vendor those files into `src/ptv3/` and expose `PointTransformerV3` through `src/ptv3/__init__.py`.


## One-time setup (vendoring the upstream files)


From the repository root:


```bash
cd /tmp
git clone https://github.com/Pointcept/PointTransformerV3.git ptv3_upstream
cd ptv3_upstream
git checkout 3229e9b7de1770c8ad17c316f8e349982de509f8

# Copy the upstream files into this workspace member.
cp model.py PUT_YOUR_DIRECTORY_HERE/TerraSeg/ptv3/src/ptv3/model.py
cp -r serialization PUT_YOUR_DIRECTORY_HERE/TerraSeg/ptv3/src/ptv3/serialization
```


After this, `PointTransformerV3` is importable from anywhere in the workspace as:


```python
from ptv3 import PointTransformerV3
```


## Attribution & License


PTv3 is copyrighted by its original authors and is distributed under the [MIT License](https://github.com/Pointcept/PointTransformerV3/blob/main/LICENSE). Vendored files retain their original copyright headers. The pinned code above is included verbatim; we make no modifications to the upstream source.
