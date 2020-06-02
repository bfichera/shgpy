#!/bin/bash
rm -r source/api
sphinx-apidoc -feT -o source/api ../shgpy
rm source/api/shgpy.rst
rm source/api/shgpy.core.rst
