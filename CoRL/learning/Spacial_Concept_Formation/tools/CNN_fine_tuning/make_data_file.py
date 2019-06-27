import glob
import os
import sys 
import numpy as np
dataset=sys.argv[1]
cd=os.getcwd()
os.chdir(dataset)
category=glob.glob("*")
category.sort()
training_num=100
train_=""
val_=""
label=0
for c in category:
    if not os.path.isdir(c):
        # ("%s is not Directory !!."%(c))
        continue
    print label,c
    file=glob.glob(c+"/*")
    file.sort()
    for i in range(len(file)):
        
        temp_=file[i]+" "+repr(label)+"\n"
        train_=train_+temp_
    #for j in range(i+1,len(file)):
        temp_=file[i]+" "+repr(label)+"\n"
        val_=val_+temp_
    label+=1
os.chdir(cd)
fw=open("train.txt","w")
fw.write(train_)
fw.close()

fw=open("val.txt","w")
fw.write(val_)
fw.close()