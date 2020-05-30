#!/bin/bash

gridsizes=(100 200 400 800)
gridlist=""
prefix="./outputs/FDM/he_outflow_tord2_sord2"
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
echo $(python ./notebooks/convergence.py -g $gridlist -p $prefix -f /output_)
