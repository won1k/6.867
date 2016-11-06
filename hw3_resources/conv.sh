#!/bin/bash

for f1 in 3 5 10
do
	for s1 in 1 2 4
	do
		for d1 in 4 16 32 64
		do
			for f2 in 3 5 10
			do
				for s2 in 1 2 4
				do
					for d2 in 4 16 32 64
					do
						for h in 16 64 128 256
						do
							for ps in 1 2
							do
								python conv.py $f1 $s1 $d1 $f2 $s2 $d2 $h $p $ps
							done
						done
					done
				done
			done
		done
	done
done