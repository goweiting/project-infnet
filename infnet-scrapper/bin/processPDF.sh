#!/bin/sh

FILES=../data/PDF/*.pdf
for f in $FILES
do
  echo "$f"
  y=${f%.pdf}
  txtName=${y##*/}
  pdf2txt.py -o ../data/pdf2txt/${txtName}.txt ${f}
done


