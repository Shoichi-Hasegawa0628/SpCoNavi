import numpy as np 
import sys

result_dir=sys.argv[1]
function_list=np.genfromtxt("/home/eagle/Spacial_Concept_Formation/parameter/description_function.txt",dtype='S')
concept_function=np.loadtxt(result_dir+"phi_w.csv")

for i in range(len(concept_function)):
    
    #print concept_name[i][0:3]
    index=np.argsort(concept_function[i])[::-1]
    concept_function[i]=np.sort(concept_function[i])[::-1]
    print "concept:"+repr(i)
    for rank in range(3):
        print repr(rank)+": "+function_list[index[rank]]+" "+repr(concept_function[i][rank])