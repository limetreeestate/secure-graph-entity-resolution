#!/bin/bash
file=$1
echo $file
gcc "${file}.cpp" -o $file -O2 -larmadillo -lstdc++ -lm
./$file