from scipy import stats
import numpy as np 
import sys 

dis1_f=sys.argv[1]
dis2_f=sys.argv[2]

dis1=np.loadtxt(dis1_f)
dis2=np.loadtxt(dis2_f)

kl=stats.entropy(dis1,dis2)

print kl