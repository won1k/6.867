#!/bin/bash

for n in 1 2 3 4
do
	for hdim in 1 10
	do
		for nlayers in 1 2
		do
			python hw3_1_4.py $n $nlayers $hdim
		done
	done
done