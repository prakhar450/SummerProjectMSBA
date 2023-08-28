#!/bin/bash

clear


echo 'Arguments to this code'

echo '1. model name to be loaded'
echo '2. folder name (beautiful etc)'
echo '3. US/OECD'
echo '4. full/small'
echo '5. output csv file name'
echo '6. batch size'

read -p 'saved_model_name_with_pt: ' model_name
read -p 'folder name (beautiful, safe, lively, depressing, boring, wealthier): ' fname
read -p 'US/OECD: ' cflag
read -p 'full/small: ' data_size
read -p 'suffix for output csv (predictedVsActualScores_suffix): ' outputcsvsuffix
read -p 'batch size: ' bsize

python3 -u Load_classification_model.py $model_name $fname $cflag $data_size $outputcsvsuffix $bsize
