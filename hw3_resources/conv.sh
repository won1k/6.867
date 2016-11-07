#!/bin/bash

for f1 in 5 7 10
do
	for s1 in 1 2 3
	do
		for d1 in 32 64 128
		do
			for f2 in 5 7 10
			do
				for s2 in 1 2 3
				do
					for d2 in 32 64 128
					do
						for h in 32 64 128
						do
							for ps in 1 2
							do
								python conv.py $f1 $s1 $d1 $f2 $s2 $d2 $h $ps
							done
						done
					done
				done
			done
		done
	done
done