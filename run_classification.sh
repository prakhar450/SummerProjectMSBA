#!/bin/bash

clear


echo 'Arguments to this code'

echo '1. model name to be saved'
echo '2. folder name (beautiful etc)'
echo '3. US/OECD'
echo '4. full/small'
echo '5. batch size'
echo '6. epochs'
echo '7. output csv file name'


read -p 'saved_model_name_with_pt: ' model_name
read -p 'folder name (beautiful, safe, lively, depressing, boring, wealthier): ' fname
read -p 'US/OECD: ' cflag
read -p 'full/small: ' data_size
read -p 'batch size: ' bsize
read -p 'number of epochs: ' nepoch
read -p 'suffix for output csv (predictedVsActualScores_suffix): ' outputcsvsuffix

nohup python3 -u Classification_SL_Resnet.py $model_name $fname $cflag $data_size $bsize $nepoch $outputcsvsuffix > output_csvs/$fname/"$cflag""_output.txt" &

#python3 -u Classification_SL_Resnet.py abcde.pt false small 5 32 1 test2 
#nohup python3 -u Classification_SL_Resnet.py abcde.pt false small 10101010101010101010 32 1 test2 &
