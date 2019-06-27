#!/usr/bin/env python
# -*- coding:utf-8 -*-
#学習したパラメータからデータをサンプリングするプログラム
#作成者　石伏智
#作成日 2015年12月
import numpy as np
import random
import sys
import re
import glob
import math
import os
import shutil
sys.path.append("../lib/")
import file_read as f_r
diric=sys.argv[1]

env_para=np.genfromtxt(diric+"/Parameter.txt",dtype= None,delimiter =" ")

MAP_X =env_para[0][1]  #地図のxの最大値
MAP_Y =env_para[1][1]  #地図のyの最大値
map_x =env_para[2][1] #地図のxの最小座標
map_y =env_para[3][1] #地図のyの最小座標

CLASS_NUM=int(env_para[4][1])
SAMPLE_NUM=500

if __name__ == '__main__':
    all_mu=f_r.mu_read(diric)#read_gauss_mu()
    all_sigma=f_r.sigma_read(diric)

    #print all_sigma[0]
    os.mkdir(diric+'/sampling_data')
    for c in range(CLASS_NUM):
        print 'class '+repr(c)+" sampling ...."

        os.mkdir(diric+'/sampling_data/class'+repr(c))
        
        for i in range(SAMPLE_NUM):
            jadge=0
            k=0
            while jadge!=1 and k!=50:
                sample=np.random.multivariate_normal(all_mu[c],all_sigma[c])
                k +=1
                if sample[0]<MAP_X and sample[0] > map_x and sample[1] < MAP_Y and sample[1] > map_y :
                    jadge=1
            
            #print sample
            f = open(diric+'/sampling_data/class'+repr(c)+'/'+repr(i)+'.txt','w')
            f.write(repr(sample[0])+','+repr(sample[1])+','+repr(sample[2])+','+repr(sample[3]))
            f.close()
    