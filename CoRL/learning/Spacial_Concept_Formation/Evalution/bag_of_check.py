import numpy as np
import sys

file=sys.argv[1]
k=int(sys.argv[2])
if k==0:
    file=file+"/bag_of_function.txt"
    list_="Spacial_Concept_Formation/parameter/description_function.txt"
else:
    file=file+"/bag_of_name.txt"
    list_="Spacial_Concept_Formation/parameter/place_name.txt"
#list_=sys.argv[2]
c=int(sys.argv[3])
data=np.loadtxt(file)
n=np.genfromtxt(list_,dtype='S')
for i in range(len(n)):
    print n[i],data[c][i]