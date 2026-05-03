#!/bin/bash

mkdir -p data/musicbrainz

wget -O data/musicbrainz/musicbrainz-20-A01.csv.dapo "https://dbs.uni-leipzig.de/files/datasets/saeedi/musicbrainz-20-A01.csv.dapo"

mv data/musicbrainz/musicbrainz-20-A01.csv.dapo data/musicbrainz/musicbrainz-20-A01.csv