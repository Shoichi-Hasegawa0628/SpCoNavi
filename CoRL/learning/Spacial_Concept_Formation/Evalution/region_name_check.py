import numpy as np 
import sys

result_dir=sys.argv[1]
region_num=int(sys.argv[2])
name_list=np.genfromtxt("/home/eagle/Spacial_Concept_Formation/parameter/place_name.txt",dtype='S')
function_list=np.genfromtxt("/home/eagle/Spacial_Concept_Formation/parameter/description_function.txt",dtype='S')
pi=np.loadtxt(result_dir+"pi.csv")
mutual_info_n=np.loadtxt(result_dir+"/mutual_info.csv")
mutual_info_w=np.loadtxt(result_dir+"/mutual_info_function.csv")
phi_n=np.loadtxt(result_dir+"/../phi_n.csv")
phi_w=np.loadtxt(result_dir+"/../phi_w.csv")

#prob_region_name=np.zeros(len(name_list))
#prob_region_function=np.zeros(len(function_list))

print len(mutual_info_n[0])
phi_n =phi_n*mutual_info_n
prob_region_name=pi.T.dot(phi_n)


phi_w =phi_w*mutual_info_w#*mutual_info_w
prob_region_function=pi.T.dot(phi_w)


index=np.argsort(prob_region_name[region_num])[::-1]

region_name=np.sort(prob_region_name[region_num])[::-1]
#region_name/=sum(region_name)

print "region:"+repr(region_num)
for rank in range(5):
    print repr(rank)+": "+name_list[index[rank]]+" "+repr(region_name[rank])
print '\n'
index=np.argsort(prob_region_function[region_num])[::-1]
region_function=np.sort(prob_region_function[region_num])[::-1]
#region_function/=sum(region_function)
for rank in range(5):
    print repr(rank)+": "+function_list[index[rank]]+" "+repr(region_function[rank])