#!/bin/bash

python setup.py sdist bdist_wheel
python -m twine upload dist/*
rm -r build
rm -r dist
rm -r shgpy.egg-info
