#!/bin/bash

clear


echo 'Arguments to this code'

echo '1. model name to be saved'

read -p 'saved_model_name_with_pt: ' model_name

nohup python3 -u main.py $model_name > nohup_output/"balanced_output.txt" &


#python3 -u Classification_SL_Resnet.py abcde.pt false small 5 32 1 test2 
#nohup python3 -u Classification_SL_Resnet.py abcde.pt false small 10101010101010101010 32 1 test2 &
