#!/bin/bash

for hdim in 1 10 50 100 500 1000
do
	for nlayers in 1 2
	do
		python hw3_1_5.py $nlayers $hdim
	done
done