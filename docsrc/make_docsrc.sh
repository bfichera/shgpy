#!/bin/bash

# This file is meant to be run in the docsrc directory.

rm -r build
rm -r source/api
sphinx-apidoc -feT -o source/api ../shgpy
rm source/api/shgpy.rst
rm source/api/shgpy.core.rst
tar -czvf source/examples.tar.gz ../examples
make html
cp source/logo.png build/html/_static/logo.png
