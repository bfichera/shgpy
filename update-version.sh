#!/bin/bash

echo $1 > .version
echo -ne "from .core import *\n__version__ = '$1'\n" > shgpy/__init__.py

rm -r build
rm -r dist
rm -r shgpy.egg-info

python setup.py sdist bdist_wheel
python -m twine upload dist/*

cd docsrc
./make_docsrc.sh
./make_ghpages.sh
