#!/bin/bash

for d1 in 4 8 16 32 64
do
	for d2 in 4 8 16 32 64
	do
		python conv.py 7 2 $d1 7 2 $d2 64
	done
done