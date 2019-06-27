import numpy as np
import sys 
description_function_list=np.genfromtxt("/home/eagle/Spacial_Concept_Formation/parameter/description_function.txt", delimiter="\n", dtype='S' )
name_list=np.genfromtxt("/home/eagle/Spacial_Concept_Formation/parameter/place_name.txt", dtype='S' )

Parameter_diric=sys.argv[1]
r=int(sys.argv[2])
mute=np.loadtxt(Parameter_diric+"/mutual_info.csv")
parameter=np.loadtxt(Parameter_diric+"/region_name/region_name"+repr(r)+".txt")
pi=np.loadtxt(Parameter_diric+"/pi.csv")
phi_n=np.loadtxt(Parameter_diric+"../phi_n.csv")
name_prob=np.zeros(len(name_list))

for w in range(len(name_list)):
    for i in range(len(pi)):
        
        name_prob[w]+=pi[i][r]*phi_n[i][w]*mute[i][w]

index=np.argsort(name_prob)[::-1]
name_prob=np.sort(name_prob)[::-1]

for rank in range(5):
    print repr(rank)+": "+name_list[index[rank]]+" "+repr(name_prob[rank])+"\n"