#!/bin/bash

gridsizes=( 50 100 200 400 800 1600)
gridlist=""
prefix="./outputs/convergence_tests/tilt45_infinite"
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

echo $(python ./notebooks/convergence.py -g $gridlist -p $prefix -f /output_)
