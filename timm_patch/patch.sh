#!/bin/bash

python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
timm_path=$CONDA_PREFIX/lib/python$python_version/site-packages/timm/models/

cp timm_patch/swin_transformer.py $timm_path
cp timm_patch/helpers.py $timm_path/layers/