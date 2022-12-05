#!/bin/bash

echo $1 > .version
echo -ne "from .core import *\n__version__ = '$1'\n" > shgpy/__init__.py
