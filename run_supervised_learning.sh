#!/bin/bash

clear

read -p 'saved_model_name_with_pth: ' model_name
read -p 'pretrained (true/false): ' pretr
read -p 'small or full: ' csvname
read -p 'number of epochs: ' nepoch
read -p 'batch size: ' bsize

python3 SL_Resnet.py $model_name $pretr $csvname $nepoch $bsize
