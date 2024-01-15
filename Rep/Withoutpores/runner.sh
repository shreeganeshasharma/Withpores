#!/bin/bash
#Aim : Run stratonnovich files without pores
#Finds MFPT(s) for various pore number and sizes
#Convention : Stratonovich
#D1/D2 = 1/1, nump = 10,000, R1 = 0.5, R2 = 1, R = R1/5 = 0.1 or 0.3, r0 = [0.2 or 0.4 or 0.75, 0. 0]
#Author: Shree Ganesha Sharma M S
#Date: November 21st 2023

declare -a filenames=("Ito_0001_11_04.py" "Iso_0001_101_04.py" "Ito_0001_110_04.py" "Strat_0001_11_04.py" "Strat_0001_101_04.py" "Strat_0001_110_04.py" "Iso_0001_11_04.py" "Iso_0001_110_04.py" "Ito_0001_101_04.py")

for value in "${!filenames[@]}"
do
	echo >> logger.txt
	echo ${filenames[value]}
	python3 ${filenames[value]} >> logger.txt
 	echo >> logger.txt
done
