#!/bin/bash

set -e

echo "Installing external repositories..."

mkdir src

git clone -b supcol https://github.com/yongrong-qiu/sensorium_2023.git src/sensorium_2023
git clone -b supcol https://github.com/yongrong-qiu/neuralpredictors.git src/neuralpredictors
git clone https://github.com/yongrong-qiu/nnfabrik.git src/nnfabrik
git clone -b supcol https://github.com/yongrong-qiu/mei.git src/mei

# pip install -e src/sensorium_2023

echo "Dependencies installed."