#!/bin/bash

gridsizes=(16 32 64 128 256 512 1024)
gridlist=""
prefix="./outputs/flatdisk_tests/dc_s2_t2"
mkdir $prefix > /dev/null 2>&1

for i in "${gridsizes[@]}"
do
	path="$prefix/n$i"
	mkdir $path > /dev/null 2>&1
	path="$path/output_"
	echo "Running n$i..."
	echo $(python params_cmd.py -n $i -f $path)
	echo "Finishing n$i..."
	if [ "$gridlist" = "" ]; then
		gridlist="$i"
	else
		gridlist="$gridlist,$i"
	fi
done

echo $prefix
echo $(python ./notebooks/convergence_simple.py -g $gridlist -p $prefix -f /output_)
echo $(python ./notebooks/convergence_simple_integrate.py -g $gridlist -p $prefix -f /output_)
