#!/bin/sh

FILES=./PDF/*.pdf
for f in $FILES
do
  echo "$f"
  y=${f%.pdf}
  txtName=${y##*/}
  pdf2txt.py -o ./txt/${txtName}.txt ${f}
done


