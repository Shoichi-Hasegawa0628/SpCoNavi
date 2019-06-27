import numpy as np 
import sys

result_dir=sys.argv[1]
name_list=np.genfromtxt("/home/eagle/Spacial_Concept_Formation/parameter/place_name.txt",dtype='S')
concept_name=np.loadtxt(result_dir+"phi_n.csv")

for i in range(len(concept_name)):
    
    #print concept_name[i][0:3]
    index=np.argsort(concept_name[i])[::-1]
    concept_name[i]=np.sort(concept_name[i])[::-1]
    print "concept:"+repr(i)
    for rank in range(3):
        print repr(rank)+": "+name_list[index[rank]]+" "+repr(concept_name[i][rank])