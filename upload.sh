#!/bin/bash

rm -r build
rm -r dist
rm -r shgpy.egg-info

python setup.py sdist bdist_wheel
python -m twine upload dist/*
