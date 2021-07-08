#!/usr/bin/env bash

set -euxo pipefail

for fn in *.ipynb ; do 
  jupyter nbconvert --execute --to html ${fn}
done
