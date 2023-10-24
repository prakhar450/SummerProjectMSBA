#!/bin/bash

clear


echo 'Arguments to this code'

echo '1. model name to be saved'
echo '2. model to be trained - resnet50, alexnetSVM'

#read -p 'saved_model_name_with_pt: ' saved_model_name
#read -p 'model_name: ' model_name

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

nohup python3 -u main.py 'alexnet_temp_model_oct20.pt' 'alexnetSVM' > nohup_output/"balanced_output.txt" &
#nohup python3 -u main.py $saved_model_name $model_name > nohup_output/"balanced_output.txt" &


#python3 -u Classification_SL_Resnet.py abcde.pt false small 5 32 1 test2 
#nohup python3 -u Classification_SL_Resnet.py abcde.pt false small 10101010101010101010 32 1 test2 &
