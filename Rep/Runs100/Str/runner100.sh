#!/bin/bash
#Aim : Run stratonnovich files with pores
#Finds MFPT(s) for various pore number and sizes
#Convention : Stratonovich
#D1/D2 = 1/1, nump = 10,000, R1 = 0.5, R2 = 1, R = R1/5 = 0.1 or 0.3, r0 = [0.2 or 0.4 or 0.75, 0. 0]
#Author: Shree Ganesha Sharma M S
#Date: November 21st 2023

declare -a filenames=("3dWithPores_Str_100_01_11.py" "3dWithPores_Str_100_03_11.py" "3dWithPores_Str_100_07_11.py" "3dWithPores_Str_100_01_110.py" "3dWithPores_Str_100_03_110.py" "3dWithPores_Str_100_07_110.py" "3dWithPores_Str_100_01_101.py" "3dWithPores_Str_100_03_101.py" "3dWithPores_Str_100_07_101.py")

for value in "${!filenames[@]}"
do
	echo >> logger100.txt
	echo ${filenames[value]}
	python3 ${filenames[value]} >> logger100.txt
 	echo >> logger100.txt
done
