# Code
To download the data, user needs to sign an agreement sheet on (https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection) and sent to ming.cheng@dukekunshan.edu.cn. After downloading the data, user can transfer them onto google VM instances and unzip them on Virtual machine with 7za x data_directory

Then user can run the preprocess.py code to transfer each video into npy files into a target directory with different resize shape depend on user machine's memory. 

After that, User can run any model to predict the violence of the data. 
