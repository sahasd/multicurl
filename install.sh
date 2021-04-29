#!/bin/bash
echo y | python3 -m pip uninstall torch
echo y | python3 -m pip uninstall torchvision
python3 -m pip install torch --no-cache-dir
python3 -m pip install torchvision --no-cache-dir
python3 -m pip install kornia --no-cache-dir
python3 -m pip install atari-py --no-cache-dir
python3 -m pip install tqdm --no-cache-dir
python3 -m pip install plotly --no-cache-dir
python3 -m pip install pettingzoo[all] --no-cache-dir 
python3 -m pip install array2gif --no-cache-dir 
python3 -m pip install supersuit --no-cache-dir 
python3 -m pip install autorom --no-cache-dir
python3 -m pip install ale-py --no-cache-dir
AutoROM