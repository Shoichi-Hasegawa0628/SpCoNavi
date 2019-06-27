import numpy as np 
import glob 
import sys
import re
diric=sys.argv[1]
file = glob.glob(diric+"/*")
convert = lambda text: int(text) if text.isdigit() else text 
alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
file.sort(key=alphanum_key)

print file

k=np.array([0 for i in range(1000)])
for f in file:

    f=np.loadtxt(f)
    f=f[:,1]
    k=k+f
    print k
diric=diric+"/../all.csv"
np.savetxt(diric,k)