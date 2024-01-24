#!/bin/bash
#Aim : Run Ito files with pores
#Finds MFPT(s) for constant porous fraction
#Convention : Ito
#D1/D2 = 1/1, nump = 10,000, R1 = 0.5, R2 = 1, R = 0.3, r0 = 0.4
#Author: Shree Ganesha Sharma M S
#Date: January 23rd 2023

declare -a filenames=("3dWithPores_Ito_100_101.py" "3dWithPores_Ito_100_110.py" "3dWithPores_Ito_100_11.py" "3dWithPores_Ito_10_101.py" "3dWithPores_Ito_10_110.py" "3dWithPores_Ito_10_11.py" "3dWithPores_Ito_20_101.py" "3dWithPores_Ito_20_110.py" "3dWithPores_Ito_20_11.py" "3dWithPores_Ito_50_101.py" "3dWithPores_Ito_50_110.py" "3dWithPores_Ito_50_11.py")


for value in "${!filenames[@]}"
do
	echo >> loggerIto.txt
	echo ${filenames[value]}
	python3 ${filenames[value]} >> loggerIto.txt
 	echo >> loggerIto.txt
done
