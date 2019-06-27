import numpy as np 
import os
import sys
import math
from numpy.linalg import inv, cholesky
import scipy.stats as ss

description_function_list=np.genfromtxt("/home/eagle/Spacial_Concept_Formation/parameter/description_function.txt", delimiter="\n", dtype='S' )
name_list=np.genfromtxt("/home/eagle/Spacial_Concept_Formation/parameter/place_name.txt", dtype='S' )




def Description_Function_data_read(directory,word_increment,DATA_NUM,test_num):
    function_data_set=[]
    for i in range(DATA_NUM):
        function_data=[0 for w in range(len(description_function_list))]
        if  (i in test_num)==False:
            try:
                file=directory+Function_data_dir+repr(i)+".txt"
                data=np.genfromtxt(file, delimiter="\n", dtype='S' )
                #print file
                
                try:
                    for d in data:
                        #print d
                        for w,dictionry in enumerate(description_function_list):
                            if d == dictionry:
                                function_data[w]+=word_increment


                except TypeError:
                    #print d
                    for w,dictionry in enumerate(description_function_list):
                        if data == dictionry:
                            function_data[w]+=word_increment
                function_data=np.array(function_data)
                function_data_set.append(function_data)
            except IOError:
                function_data=np.array(function_data)
                function_data_set.append(function_data)
    return np.array(function_data_set)

def Name_data_read(directory,word_increment,DATA_NUM):
    name_data_set=[]
    
    for i in range(DATA_NUM):
        name_data=[0 for w in range(len(name_list))]

            
        try:
            file=directory+repr(i)+".txt"
            #print file
            data=np.genfromtxt(file, delimiter="\n", dtype='S' )
            #print file

            try:
                for d in data:
                    #print d
                    for w,dictionry in enumerate(name_list):
                        if d == dictionry:
                            name_data[w]+=word_increment


            except TypeError:
                #print d
                for w,dictionry in enumerate(name_list):
                    if data == dictionry:
                        name_data[w]+=word_increment
        except IOError:
            
            pass
        name_data=np.array(name_data)
        name_data_set.append(name_data)

        #else:
            #print i,"is test data."
    return np.array(name_data_set)

train_dir=sys.argv[1]
data_num=int(sys.argv[2])
print data_num
name_dir=train_dir+"/name/"
name_vector= Name_data_read(name_dir,1,data_num)
sum_n_vector=sum(name_vector)

function_dir=train_dir+"/function/"
function_vector= Name_data_read(function_dir,1,data_num)
sum_f_vector=sum(name_vector)
file=open(train_dir+"name_count.txt",'w')
for n in range(len(sum_n_vector)):
    file.write(name_list[n]+","+repr(sum_n_vector[n])+"\n")
file.close()


file=open(train_dir+"function_count.txt",'w')
for n in range(len(sum_f_vector)):
    file.write(description_function_list[n]+": "+repr(sum_f_vector[n])+"\n")
file.close()

