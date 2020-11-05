#!/bin/bash
file=$1
echo $file
gcc "${file}.cpp" MurmurHash3.cpp -o $file -O2 -larmadillo -lstdc++ -lm
./$file