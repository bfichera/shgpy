#!/bin/bash
rm -r ../docs/*
cp -r build/html/* ../docs
touch ../docs/.nojekyll
