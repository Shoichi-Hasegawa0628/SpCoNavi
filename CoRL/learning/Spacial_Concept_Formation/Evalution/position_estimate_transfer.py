#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np 
import argparse
import os
import sys
import math
from numpy.linalg import inv, cholesky
import glob
import re
import shutil
import pypr.clustering.gmm as gmm
description_function_list=np.genfromtxt("/home/eagle/Spacial_Concept_Formation/parameter/description_function.txt", delimiter="\n", dtype='S' )
name_list=np.genfromtxt("/home/eagle/Spacial_Concept_Formation/parameter/place_name.txt", dtype='S' )

parser = argparse.ArgumentParser()
parser.add_argument(
    "parameter_dir",
    help="Input training directory."
)

args = parser.parse_args()


def Description_Function_data_read(data,word_increment):

    function_data=[0 for w in range(len(description_function_list))]
    
        
    #data=np.genfromtxt(file, delimiter="\n", dtype='S' )
            #print file
            
    try:
        for d in data:
                    #print d
            for w,dictionry in enumerate(description_function_list):
                if d == dictionry:
                    function_data[w]+=word_increment


    except TypeError:
            #print d
        for dictionry in description_function_list:
            if data == dictionry:
                function_data[w]+=word_increment
    function_data=np.array(function_data)

    return function_data

def Name_data_read(file,word_increment):

    name_data=[0 for w in range(len(name_list))]
    try:
        
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
        #name_data=np.array(name_data)
    return np.array(name_data)

def Multi_prob(data,phi):
    phi_log=np.log(phi)
    prob = data.dot(phi_log.T)
    return prob

def position_data_read(name_vector,train_dir, data_num):
    position_set=[]
    index_set=[]
    for d in range(data_num):
        file=train_dir+"/name/"+repr(d)+".txt"
        data=Name_data_read(file,20)
        check =1
        for k in range(len(name_vector)):
            if name_vector[k]>0 and name_vector[k]!=data[k]:
                check=0
        if check==1:
            pose=np.loadtxt(train_dir+"/position_data/"+repr(d)+".txt")
            test_pose=[pose[0][0],pose[0][1]]
            position_set.append(test_pose)
            index_set.append(d)
    return position_set,index_set



parameter_dir=args.parameter_dir
result_env=np.genfromtxt(parameter_dir+"/Parameter.txt",delimiter=": ",dtype='S')
train_dir=result_env[8][1]
train_env=np.genfromtxt(train_dir+"/Environment_parameter.txt",delimiter=" ",dtype='S')
data_num=int(train_env[6][1])-int(train_env[5][1])+1
print train_dir,"data_num: ",repr(data_num)

#test_dir=sys.argv[2]
try:
    phi_n=np.loadtxt(parameter_dir+"/phi_n_e.csv")
    phi_f=np.loadtxt(parameter_dir+"/phi_f_e.csv")
    phi_w=np.loadtxt(parameter_dir+"/phi_w_e.csv")
except IOError:
    phi_n=np.loadtxt(parameter_dir+"/../phi_n.csv")
    phi_f=np.loadtxt(parameter_dir+"/../phi_f.csv")
    phi_w=np.loadtxt(parameter_dir+"/../phi_w.csv")
pi=np.loadtxt(parameter_dir+"/pi.csv")
G=np.loadtxt(parameter_dir+"/G.txt")
region_num=len(pi[0])
sigma_set=[]
mu_set=[]
mutual_info=np.loadtxt(parameter_dir+"/mutual_info.csv")
for i in range(region_num):
    sigma=np.loadtxt(parameter_dir+"/sigma/gauss_sigma"+repr(i)+".csv")
    mu=np.loadtxt(parameter_dir+"/mu/gauss_mu"+repr(i)+".csv")
    sigma_set.append(sigma)
    mu_set.append(mu)
samp_num=500


Place=[["本棚前"],["キッチン"]]
#Description=[["キッチン"],["出入口"],["扉の前"],["扉の前","出入口"],["本棚前"],["ミーティングスペース"],["個人デスク"]]
#Description=[[],["扉を開ける"],["扉を開ける"],["扉を開ける"],["本を借りる"]]
Out_put_dir=parameter_dir+"/place_sampling/"
class_choice=np.array([i for i in range(len(pi))])
region_choice=np.array([i for i in range(len(pi[0]))])

#
sample_num=100
try:
    os.mkdir(Out_put_dir)                
except OSError:
    shutil.rmtree(Out_put_dir)
    os.mkdir(Out_put_dir)

for p in range(len(Place)):
    print "{0}".format(Place[p][0])
    name_vector=Name_data_read(Place[p],20)
    #description_function
    #print name_vector
    position_set,index_set=position_data_read(name_vector,train_dir,data_num)
    #print index_set
    #sys.exit()
    C_t_prob=np.array([0.0 for i in xrange(len(G))])
    #
    for c in xrange(len(G)):
        C_t_prob[c]= math.log(G[c])+Multi_prob(name_vector,phi_n[c])

    """
    for c in xrange(len(G)):
         C_t_prob[c]=math.log(G[c])+Multi_prob(name_vector,phi_n[c])
    """


    max_class=np.max(C_t_prob)
    C_t_prob -=max_class
    C_t_prob = np.exp(C_t_prob)
    C_t_prob =C_t_prob /sum(C_t_prob)
    C_t=np.argmax(C_t_prob)
    print "C_t:",C_t
    prob_rt=[0.0 for i in xrange(len(pi[0]))]
    for c in xrange(len(G)):
        for r in xrange(len(pi[0])):
            prob_rt[r] +=pi[C_t][r]
    #print prob_rt
    prob_rt /=sum(prob_rt)
    ##print prob_rt
    np.savetxt(repr(p)+".txt",prob_rt)
    #X=gmm.sample_gaussian_mixture(mu_set,sigma_set,prob_rt,samples=sample_num)
    r_t=np.argmax(prob_rt)
    print "r_t:",r_t

    X= np.random.multivariate_normal(mu_set[r_t],sigma_set[r_t],sample_num)
    all_cost=0.0
    for x in X:
        print x 
        min_cost=None
        #print position_set
        for pose in position_set:
            cost=((pose[0]-x[0])**2)+((pose[1]-x[1])**2)
            cost=cost**0.5  
            print cost
            if min_cost==None:
                min_cost=cost

            elif cost< min_cost:
                min_cost=cost
        all_cost +=min_cost
    #print repr(p)
    print "all_cost",(all_cost/sample_num)
    #print X

    os.mkdir(Out_put_dir+repr(p))

    np.savetxt(Out_put_dir+repr(p)+"/sample.csv",X)



