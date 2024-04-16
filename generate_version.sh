#!/usr/bin/env bash

TARGET_FILE=deid_doc/_version.py
export PACKAGE_VERSION=$(cat pyproject.toml | grep -e "^version" | cut -d'"' -f 2)
export GIT_HASH=$(git rev-parse --short HEAD)
export GIT_VERSION=$(git describe)

echo "# This file was generated automatically, do not edit" > $TARGET_FILE
echo "__version__ = \"$PACKAGE_VERSION\"" >> $TARGET_FILE
echo "__githash__ = \"$GIT_HASH\"" >> $TARGET_FILE
echo "__gitversion__ = \"$GIT_VERSION\"" >> $TARGET_FILE
