#!/bin/sh

#Cleans the PDF and pdf2txt folder anf remove emmpty files (size 0)
echo "Deleting files with size 0"
num=$(find ../data/pdf2txt/ -size 0 | wc -l)
echo "$num files with size 0"
echo "deleting...."
find ../data/pdf2txt -size 0 -delete

echo "DONE"
