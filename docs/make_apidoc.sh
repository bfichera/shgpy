#!/bin/bash
rm -r source/api
sphinx-apidoc -feT -o source/api ../shgpy
