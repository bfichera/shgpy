#!/bin/bash

echo $1 > .version

python setup.py sdist bdist_wheel
pip install dist/shgpy-$1.tar.gz --upgrade

rm -r build
rm -r dist
rm -r shgpy.egg-info

cd docsrc
./make_docsrc.sh
./make_ghpages.sh
