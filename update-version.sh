#!/bin/bash

echo $1 > .version

rm -r build
rm -r dist
rm -r shgpy.egg-info

python setup.py sdist bdist_wheel
python -m twine upload dist/*

cd docsrc
./make_docsrc.sh
./make_ghpages.sh
