#!/usr/bin/env bash

mkdir -p data
pushd data

curl https://data.deepai.org/mnist.zip --output mnist.zip
unzip mnist

popd
