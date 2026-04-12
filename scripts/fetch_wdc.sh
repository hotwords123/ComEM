#!/bin/bash
set -e

# Step 1: Download the zip file
mkdir -p data
curl -o data/80multi.zip https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/80multi.zip

# Step 2: Unzip to data/wdc/
mkdir -p data/wdc
unzip -o data/80multi.zip -d data/wdc/

# Step 3: Gunzip all .gz files in data/wdc/
find data/wdc/ -name "*.gz" -exec gunzip -f {} \;