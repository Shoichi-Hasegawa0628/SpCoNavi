#!/usr/bin/env python
# -*- coding:utf-8 -*-
##########################################
#Gibbs sampling for training Place concept 
#Author Satoshi Ishibushi

#-==========================================-

##########################################
import argparse
import numpy as np
import random
import string
import sys
import glob
import re
import math
import os
from numpy.linalg import inv, cholesky
from scipy.stats import chi2
import time
sys.path.append("../lib/")
import BoF
import Prob_Cal
import Multi
import file_read as f_r
import nonpara_tool
import shutil
import scipy.stats as ss
import mutual_information as mutual
from sklearn.preprocessing import normalize
#CNN_feature=1 #If you want to use image feature of 4096 dimensions,you shold set 1.
start_time=time.time()

parser = argparse.ArgumentParser()
parser.add_argument(
    "Training_dataset_list",
    help="Input training directory."
)
parser.add_argument(
    "Output",
    help="Output directory."
)
parser.add_argument(
    "--slide",
    default=1,
    help="Sliding num for reading training data."
)
parser.add_argument(
    "--feature",
    type=str,
    default=None,
    help="feature_dirc"
)
parser.add_argument(
    "--iteration",
    type=int,
    default=100,
    help="iteration"

)
parser.add_argument(
    "--word_not",
    type=int,
    default=None,
    help="If you don't use word for environmet."

)
parser.add_argument(
    "--function_not",
    type=int,
    default=None,
    help="If you don't use function description for environmet."

)
parser.add_argument(
    "--test_num",
    type=int,
    default=1,
    help="If you don't use function description for environmet."

)
args = parser.parse_args()
Traing_list=np.genfromtxt(args.Training_dataset_list , delimiter="\t", dtype='S' )
try:
    data_set_num=len(Traing_list)
except TypeError:
    Traing_list=np.array([Traing_list])
    data_set_num=1
Slide=int(args.slide)
feature_data_dir="/googlenet_prob/"
Name_data_dir="/name/"
Function_data_dir="/function/"


#===================Environment_parameter=========================
"""
#print env_para


map_center_x = ((MAP_X - map_x)/2)+map_x
map_center_y = ((MAP_Y - map_x)/2)+map_y
concept_num=env_para[4][1]
region_num=100

DATA_initial_index= int(env_para[5][1]) #Initial data num
DATA_last_index= int(env_para[6][1]) #Last data num
DATA_NUM =DATA_last_index - DATA_initial_index +1
Learnig_data_num=(DATA_last_index - DATA_initial_index +1)/Slide #Data number
""" 

hyper_para=np.loadtxt("../parameter/gibbs_trans_hyper_parameter.txt",delimiter=" #",skiprows=2)

Stick_large_L=150
Stick_large_R=80
sigma_init = 100.0 #initial covariance value
alpha_f = hyper_para[0]
alpha_w = hyper_para[1]
alpha_n = hyper_para[2]
kappa_0=hyper_para[3]
nu_0=hyper_para[4]
#mu_0 = np.array([0.0,0.0,0.0,0.0])
psai_0 = np.matrix([[0.05,0.0,0.0,0.0],[0.0,0.05,0.0,0.0],[0.0,0.0,50.0,0.0],[0.0,0.0,0.0,50.0]])
gamma = hyper_para[5]
gamma_0 = hyper_para[6]
beta = hyper_para[7]
image_increment = hyper_para[8] 
word_increment = hyper_para[9]
alpha_e=10
iteration = 1000#args.iteration 

cut_prob=0.2

test_num=args.test_num


description_function_list=np.genfromtxt("../parameter/description_function.txt", delimiter="\n", dtype='S' )
name_list=np.genfromtxt("../parameter/place_name.txt", delimiter="\n", dtype='S' )

#print "Description Function List"
#for n in range(len(description_function_list)):
#    print description_function_list[n]

#<<<<<<<<<<<<<<<<Gibbs sampling>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#===================================================================================

#data_pose,data_feature,data_word,word_data_ind

def position_data_read_pass(directory,DATA_NUM,test_num):
    all_position=[] 

    for i in range(DATA_NUM):
        if  (i in test_num)==False:
            f=directory+"position_data/"+repr(i)+".txt"
            position=[] #(x,y,sin,cos)
            for line in open(f, 'r').readlines():
                data=line[:-1].split(' ')
                position +=[float(data[0])]
                position +=[float(data[1])]
            all_position.append(position)
    
    return all_position


def feature_data_read_pass(directory,S,DATA_NUM,test_num):
    all_feature=[]

    for i in range(DATA_NUM):
        if  (i in test_num)==False:
            f=directory+repr(i)+".csv"
            feat=np.loadtxt(f)
            try:
                feat=feat[:,1]
                feat=np.array(feat)

            except IndexError:
                #feat=np.loadtxt(f)
                pass
            
            feat =feat*S
            all_feature.append(feat)
        #print f
    return all_feature


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

def Name_data_read(directory,word_increment,DATA_NUM,test_num):
    name_data_set=[]
    
    for i in range(DATA_NUM):
        name_data=[0 for w in range(len(name_list))]

        if  (i in test_num)==False:
            try:
                file=directory+Name_data_dir+repr(i)+".txt"
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
        else:
            print i
        #else:
            #print i,"is test data."
    return np.array(name_data_set)

def Name_data_read_cut(directory,word_increment,DATA_NUM,test_num):
    name_data_set=[]
    cut=cut_prob
    for i in range(DATA_NUM):
        name_data=[0 for w in range(len(name_list))]
        if  (i in test_num)==False:
            try:
                file=directory+Name_data_dir+repr(i)+".txt"
                data=np.genfromtxt(file, delimiter="\n", dtype='S' )
                #print file
                if cut==1.0:
                    increment=word_increment
                    cut=cut_prob
                else:
                    increment=0.0
                    cut +=cut_prob
                

                try:
                    for d in data:
                        #print d
                        for w,dictionry in enumerate(name_list):
                            if d == dictionry:
                                name_data[w]+=increment


                except TypeError:
                    #print d
                    for w,dictionry in enumerate(name_list):
                        if data == dictionry:
                            name_data[w]+=increment
            except IOError:
                pass
            name_data=np.array(name_data)
            name_data_set.append(name_data)
        #else:
            #print i,"is test data."
    return np.array(name_data_set)

def Multi_prob(data,phi):
    phi_log=np.log(phi)
    prob = data.dot(phi_log.T)
    return prob

def test_data_read(file,test_num):
    i=0
    test_data_num=[]
    for line in open(file, 'r').readlines():
        
        if i==test_num:
            num=line[:-1].split(',')
            
            for n in num:
                try:
                    test_data_num.append(int(n))
                except ValueError:
                    pass
        i+=1
    return test_data_num
def gibbs():

#======================================Reading Data===========================================
    print "Start reading Data"

    feature_set=[]
    feature_factorial_set=[]
    position_set=[]
    function_set=[]
    name_set=[]
    C_t_set=[]
    r_t_set=[]
    
    G_set=[]
    pi_set=[]
    Sigma_set=[]
    Mu_set=[]

    data_num=[]
    G_0=nonpara_tool.stick_breaking(gamma_0,Stick_large_L)
    concept_num=len(G_0)
    region_num=Stick_large_R
    MAP_X=[0 for e in range(data_set_num)]
    MAP_Y=[0 for e in range(data_set_num)]
    map_x=[0 for e in range(data_set_num)]
    map_y=[0 for e in range(data_set_num)]
    mu_0_set=[]
    for e,dir in enumerate(Traing_list):
        print "Reading ",dir," ..."

        test_data_list=dir+"/test_num.txt"
        test_data_num=test_data_read(test_data_list,test_num)
        #print test_data_num

        #if args.feature==None: 
        feature_dir = dir+feature_data_dir
        env_para = np.genfromtxt(dir+"/Environment_parameter.txt",dtype= None,delimiter =" ")

        MAP_X[e] = float(env_para[0][1])  #Max x value of the map
        MAP_Y[e] = float(env_para[1][1])  #Max y value of the map
        map_x[e] = float(env_para[2][1]) #Min x value of the map
        map_y[e] = float(env_para[3][1]) #Max y value of the map

        map_center_x = ((MAP_X[e] - map_x[e])/2)+map_x[e]
        map_center_y = ((MAP_Y[e] - map_x[e])/2)+map_y[e]
        mu_0=np.array([map_center_x,map_center_y,0,0])
        mu_0_set.append(mu_0)
        DATA_initial_index = int(env_para[5][1]) #Initial data num
        DATA_last_index = int(env_para[6][1]) #Last data num
        DATA_NUM =DATA_last_index - DATA_initial_index +1


        pose=np.array(position_data_read_pass(dir,DATA_NUM,test_data_num))
        position_set.append(pose)

        feature=np.array(feature_data_read_pass(feature_dir,image_increment,DATA_NUM,test_data_num))
        feature_set.append(feature)

        #print feature[0]
        if args.function_not==e:
            print "In environment "+repr(e)+",word data is not used."
            function=[]
            for d in range(DATA_NUM):
                function_d=np.array([0.0 for k in range(len(description_function_list))])
                function.append(function_d)
        else:
            function=Description_Function_data_read(dir,word_increment,DATA_NUM,test_data_num)
        function_set.append(function)

        if args.word_not==e:
            print "In environment "+repr(e)+",word data is not used."
            name=Name_data_read_cut(dir,word_increment,DATA_NUM,test_data_num)

        else:
            name=Name_data_read(dir,word_increment,DATA_NUM,test_data_num)
        print sum(name)
        name_set.append(name)

        DATA_NUM=DATA_NUM-len(test_data_num)
        Learnig_data_num=(DATA_last_index - DATA_initial_index +1)/Slide #Data number
        data_num.append(DATA_NUM)
        #word,word_data_ind = f_r.word_data_read(data_diric,DATA_NUM,space_name,DATA_initial_index,word_class,word_increment,word_diric)
        #word = f_r.word_data_read(data_diric,DATA_NUM,space_name,DATA_initial_index,word_class,word_increment,word_diric)
        G=np.random.dirichlet(G_0+gamma)
        G_set.append(G)

        pi_e=[]
        for i in range(concept_num):
                pi=nonpara_tool.stick_breaking(beta,Stick_large_R)
                pi_e.append(pi)
        pi_set.append(pi_e)

        c_t=[1000 for n in xrange(DATA_NUM)] 
        C_t_set.append(c_t)

        r_t=[1000 for n in xrange(DATA_NUM)]
        r_t_set.append(r_t)

        Mu= []       
        for j in range(region_num):
            index=int(random.uniform(0,DATA_NUM))
            p=pose[index]
            Mu.append(p)
        Mu=np.array(Mu)
        Mu_set.append(Mu)

        #Sigma = [np.matrix([[sigma_init,0.0,0.0,0.0],[0.0,sigma_init,0.0,0.0],[0.0,0.0,sigma_init,0.0],[0.0,0.0,0.0,sigma_init]]) for i in range(region_num)]
        #Sigma = np.array(Sigma)

        Sigma = np.array([[[sigma_init,0.0,0.0,0.0],[0.0,sigma_init,0.0,0.0],[0.0,0.0,sigma_init,0.0],[0.0,0.0,0.0,sigma_init]] for i in range(region_num)])
        Sigma_set.append(Sigma)



    #Initializing class index of 
    #print len(feature_set[0][0])
    FEATURE_DIM = len(feature_set[0][0])
    FUNCTION_DIM =len(function_set[0][0])
    NAME_DIM = len(name_set[0][0])
    phi_f_e = np.array([[[float(1.0)/FEATURE_DIM for i in range(FEATURE_DIM)] for j in range(concept_num)] for k in range(data_set_num)])

        #Initializing the mean of Multinomial distribution for Words
    phi_w_e=np.array([[[float(1.0)/FUNCTION_DIM for i in range(FUNCTION_DIM)] for j in range(concept_num)] for k in range(data_set_num)])

        #Initializing the mean of Multinomial distribution for name
    phi_n_e=np.array([[[float(1.0)/NAME_DIM for i in range(NAME_DIM)] for j in range(concept_num)] for k in range(data_set_num)])
    #Initializing the mean of Multinomial distribution for Image features
    phi_f = np.array([[float(1.0)/FEATURE_DIM for i in range(FEATURE_DIM)] for j in range(concept_num)])

    #Initializing the mean of Multinomial distribution for Words
    phi_w=np.array([[float(1.0)/FUNCTION_DIM for i in range(FUNCTION_DIM)] for j in range(concept_num)])

    #Initializing the mean of Multinomial distribution for name
    phi_n=np.array([[float(1.0)/NAME_DIM for i in range(NAME_DIM)] for j in range(concept_num)])
    region_choice=[dr for dr in range(region_num)]
    class_choice=[dc for dc in range(concept_num)]
    print data_num
    for iter in xrange(iteration):
        
        all_prob=1.0
        print "iteration"+ repr(iter)+"NEW_MODEL"
        class_count=np.array([0.0 for i in range(concept_num)])
        
        
        region_count=[[0.0 for i in range(region_num)] for j in range(data_set_num)]
        class_count_e_set=[]
        for e in range(data_set_num):
            
            class_region_count=[[0.0 for i in range(region_num)] for j in range(concept_num)]
            class_count_e=[0.0 for i in range(concept_num)]
            print 'Environment '+repr(e)+'\n'



        #============ estimating ====================================            
        #<<<<<Sampling class index C_t<<<<
            #print 'Sampling calss index...\n'
            #gauss_prob_set=[]
            gauss_prob_set=np.zeros((region_num,data_num[e]),dtype=float)
            if iter==0:
                C_t=np.random.randint(concept_num,size=data_num[e])
            else:
                C_t=C_t_set[e]
            pi_e=np.array(pi_set[e])
            #print pi_e
            #print np.log(pi_e)
            pi_data = np.array([pi_e[C_t[d]] for d in range(data_num[e])])
            pi_data =np.log(pi_data)
            #print len(pi_data),len(pi_data[0])
            for i in range(region_num):
                gauss_prob=ss.multivariate_normal.logpdf(position_set[e],Mu_set[e][i],Sigma_set[e][i])+pi_data[:,i]
                gauss_prob_set[i] +=gauss_prob
            

            gauss_prob_set=gauss_prob_set.T
            max_region =np.max(gauss_prob_set,axis=1)
            gauss_prob_set =gauss_prob_set -max_region[:,None]
            gauss_prob_set=np.exp(gauss_prob_set)
            sum_set=np.sum(gauss_prob_set,axis=1)
            gauss_prob_set=gauss_prob_set / sum_set[:,None]
            for d in xrange(0,data_num[e],Slide):
                r_t_set[e][d] =np.random.choice(region_choice,p=gauss_prob_set[d])
                region_count[e][r_t_set[e][d]] += 1.0
            r_t=r_t_set[e]

            phi_f_log=np.log(phi_f)
            phi_n_log=np.log(phi_n)
            phi_w_log=np.log(phi_w)

            multi_prob_set=np.zeros((concept_num,data_num[e]),dtype=float)
            pi_data=np.array([pi_e.T[r_t[d]] for d in range(data_num[e])])
            pi_data=np.log(pi_data)
            G_log=np.log(G_set[e])
            #print len(pi_data),len(pi_data[0])

            for i in range(concept_num):
                image_prob=feature_set[e].dot(phi_f_log[i])
                function_prob=function_set[e].dot(phi_w_log[i])
                name_prob=name_set[e].dot(phi_n_log[i])
                modal_prob=image_prob+name_prob+function_prob+pi_data[:,i]
                #print modal_prob[0],modal_prob[5]
                modal_prob=modal_prob+G_log[i]
                #print modal_prob[0],modal_prob[5]
                #print len(modal_prob)
                multi_prob_set[i] +=modal_prob
            multi_prob_set =multi_prob_set.T
            max_concept =np.max(multi_prob_set,axis=1)
            multi_prob_set = multi_prob_set - max_concept[:,None]
            multi_prob_set =np.exp(multi_prob_set)
            sum_concept_set=np.sum(multi_prob_set ,axis=1)
            multi_prob_set = multi_prob_set / sum_concept_set[:,None]
            for d in xrange(0,data_num[e],Slide):
                C_t_set[e][d] =np.random.choice(class_choice,p=multi_prob_set[d])
                class_count_e[C_t_set[e][d]]+= 1.0
                class_region_count[C_t_set[e][d]][r_t[d]] +=1.0

        
            for r in xrange(region_num):
                pose_r=[]
                
                #========Calculating average====
                for d in xrange(data_num[e]):
                    if r_t_set[e][d]==r:
                        pose_r.append(position_set[e][d])
                        

                sum_pose=np.zeros(4)#([0.0,0.0,0.0,0.0])
                for i in xrange(len(pose_r)):
                    for j in xrange(4):
                        sum_pose[j] +=pose_r[i][j]
    			
                bar_pose=np.zeros(4)#([0.0,0.0,0.0,0.0])
                for i in xrange(4):
                    if sum_pose[i] !=0:		 	
                        bar_pose[i]=sum_pose[i]/len(pose_r)


			    #=========Calculating Mu=============
                Mu = (kappa_0*mu_0_set[e]+len(pose_r)*bar_pose)/(kappa_0+len(pose_r))

                #=========Calculating Matrix_R===================
                bar_pose_matrix=np.matrix(bar_pose)
                Matrix_R =np.zeros([4,4])
                for i in xrange(len(pose_r)):
                    pose_r_matrix = np.matrix(pose_r[i])
                    Matrix_R +=((pose_r_matrix- bar_pose_matrix).T*(pose_r_matrix- bar_pose_matrix))

                #=======Calculating Psai===============
                ans = ((bar_pose_matrix - mu_0_set[e]).T * (bar_pose_matrix - mu_0_set[e]))*((kappa_0*len(pose_r))/(kappa_0+len(pose_r)))
                Psai = psai_0 + Matrix_R + ans
    		 	
    		 	#=======Updating hyper parameter:Kappa,Nu===============================
                Kappa = kappa_0 + len(pose_r)
                Nu = nu_0 + len(pose_r)

    		 	#============Sampling fron wishrt dist====================
                Sigma_set[e][r]=Prob_Cal.sampling_invwishartrand(Nu,Psai)
                Sigma_temp=Sigma_set[e][r]/Kappa

                Mu_set[e][r]= np.random.multivariate_normal(Mu,Sigma_temp)
                #No asigned data
                if len(pose_r)==0:
                    #index=int(random.uniform(0,data_num[e]))
                    #p=position_set[e][index]
                    p=np.array([random.uniform(map_x[e],MAP_X[e]),random.uniform(map_y[e],MAP_Y[e]),random.uniform(-1.0,1.0),random.uniform(-1.0,1.0)])
                    Mu_set[e][r]=p
                    #Sigma_set[e][r]=np.matrix([[sigma_init,0.0,0.0,0.0],[0.0,sigma_init,0.0,0.0],[0.0,0.0,sigma_init,0.0],[0.0,0.0,0.0,sigma_init]])
                    Sigma_set[e][r]=np.array([[sigma_init,0.0,0.0,0.0],[0.0,sigma_init,0.0,0.0],[0.0,0.0,sigma_init,0.0],[0.0,0.0,0.0,sigma_init]])

                for c in range(concept_num):
                    pi_set[e][c] = np.random.dirichlet(class_region_count[c]+beta)
            total_feat_e_set=[]
            total_function_e_set=[]
            total_name_e_set=[]
            for c in range(concept_num):
                feat_c_e=[]
                function_c_e=[]
                name_c_e=[]
                for e in range(data_set_num):
                    for d in xrange(data_num[e]):
                        if C_t_set[e][d]==c:
                            feat_c_e.append(feature_set[e][d])
                            function_c_e.append(function_set[e][d])
                            name_c_e.append(name_set[e][d])

                    total_feat_e = BoF.bag_of_feature(feat_c_e,FEATURE_DIM)
                    total_feat_e_set.append(total_feat_e)
                    #print alpha_f*alpha_e
                    alpha_f_e=alpha_f*alpha_e
                    total_feat_e = total_feat_e+(phi_f[c]*(alpha_f_e))
                    phi_f_e[e][c] = np.random.dirichlet(total_feat_e)+1e-100


                    total_function_e=BoF.bag_of_feature(function_c_e,FUNCTION_DIM)
                    total_function_e_set.append(total_function_e)
                    alpha_w_e=alpha_w*alpha_e
                    total_function_e=total_function_e +(phi_w[c]*alpha_w_e)
                    phi_w_e[e][c]=np.random.dirichlet(total_function_e)+1e-100


                    total_name_e=BoF.bag_of_feature(name_c_e,NAME_DIM)
                    total_name_e_set.append(total_name_e)
                    alpha_n_e=alpha_n*alpha_e
                    total_name_e=total_name_e +(phi_n[c]*alpha_n_e)
                    phi_n_e[e][c]=np.random.dirichlet(total_name_e)+1e-100
            #class_count_e=class_count_e*G_0
            G_set[e]=np.random.dirichlet(class_count_e+(gamma*G_0))+1e-100
            class_count_e_set.append(class_count_e)
                #print np.random.dirichlet(G_0*class_count+gamma)
        #print 'Finished sampling parameters of Position Gaussian dist...\n'
	
        #<<<<<<Sampling Parameter General Place Concept>>>>>>>>>>>>>>>>>
        total_feat_set=[]
        total_function_set=[]
        total_name_set=[]
        class_count=[0.0 for i in range(concept_num)]
        #print 'Started sampling parameters of General Place Concept'
        
        for c in xrange(concept_num):
            feat_c=[]
            function_c=[]
            name_c=[]
            for e in range(data_set_num):
                for d in xrange(data_num[e]):
                    if C_t_set[e][d]==c:
                        feat_c.append(feature_set[e][d])
                        function_c.append(function_set[e][d])
                        name_c.append(name_set[e][d])
                        class_count[c] += 1.0

            total_feat = BoF.bag_of_feature(feat_c,FEATURE_DIM)
            total_feat_set.append(total_feat)
            total_feat = total_feat+alpha_f
            phi_f[c] = np.random.dirichlet(total_feat)+1e-100


            total_function=BoF.bag_of_feature(function_c,FUNCTION_DIM)
            total_function_set.append(total_function)
            total_function=total_function +alpha_w
            phi_w[c]=np.random.dirichlet(total_function)+1e-100


            total_name=BoF.bag_of_feature(name_c,NAME_DIM)
            total_name_set.append(total_name)
            total_name=total_name +alpha_n
            phi_n[c]=np.random.dirichlet(total_name)+1e-100

            class_count[c]+= gamma_0
            

            if len(feat_c)==0:

                phi_f[c]=np.array([float(1.0)/FEATURE_DIM for i in range(FEATURE_DIM)])
                phi_w[c]=np.array([float(1.0)/FUNCTION_DIM for i in range(FUNCTION_DIM)] )
                phi_n[c]=np.array([float(1.0)/NAME_DIM for i in range(NAME_DIM)] )

        G_0=np.random.dirichlet(class_count)

        #print 'Finished sampling parameters of index Multinomial dist...\n'
        #f_prob.write(repr(all_prob)+"\n")
        #f_prob.close()
        print 'Iteration ',iter+1,' Done..\n'

        if (iter%50)==0:
            C_num=0
            for i in range(concept_num):
                if class_count[i]>gamma_0:
                    C_num +=1


            print "Class num:"+repr(C_num)+"\n"
            print region_count
            R_set=[0.0 for e in range(data_set_num)]
            for e in range(data_set_num):
                for r in range(region_num):
                    if region_count[e][r]>0:
                        R_set[e] +=1.0
            #================Saving=====================
            
            Out_put_dir="../result/"+args.Output+"_iter_"+repr(iter)
            try:
                os.mkdir(Out_put_dir)
            except OSError:
                shutil.rmtree(Out_put_dir)
                os.mkdir(Out_put_dir)
            f=open("training dataset",'w')
            for i,d in enumerate(Traing_list):
                w=repr(i)+":  "+d
                f.write(w)
            f.close()
            #saving finish time
            finish_time=time.time()- start_time
            f=open(Out_put_dir+"/time.txt","w")
            f.write("time:"+repr(finish_time)+" seconds.")
            f.close()

            f=open(Out_put_dir+"/training dataset",'w')
            for i,d in enumerate(Traing_list):
                w=repr(i)+":  "+d+"\n"
                f.write(w)
            f.close()

            #==================================================================
            #====================saving environment parameter================================
            for i in range(data_set_num):
                os.mkdir(Out_put_dir+"/dataset"+repr(i))
                os.mkdir(Out_put_dir+"/dataset"+repr(i)+"/mu")
                os.mkdir(Out_put_dir+"/dataset"+repr(i)+"/sigma")
                #os.mkdir(Out_put_dir+"/dataclass")
                os.mkdir(Out_put_dir+"/dataset"+repr(i)+"/region_name")
                os.mkdir(Out_put_dir+"/dataset"+repr(i)+"/region_function")
                np.savetxt(Out_put_dir+"/dataset"+repr(i)+"/class.txt",C_t_set[i])
                np.savetxt(Out_put_dir+"/dataset"+repr(i)+"/region.txt",r_t_set[i])
                np.savetxt(Out_put_dir+"/dataset"+repr(i)+"/region_count.txt",region_count[i])
                np.savetxt(Out_put_dir+"/dataset"+repr(i)+"/pi.csv",pi_set[i])
                np.savetxt(Out_put_dir+"/dataset"+repr(i)+"/G.txt",G_set[i])
                np.savetxt(Out_put_dir+"/dataset"+repr(i)+"/class_count_e.txt",class_count_e_set[e])
                np.savetxt(Out_put_dir+"/dataset"+repr(i)+"/phi_f_e.csv",phi_f_e[i])
                np.savetxt(Out_put_dir+"/dataset"+repr(i)+"/phi_w_e.csv",phi_w_e[i])
                np.savetxt(Out_put_dir+"/dataset"+repr(i)+"/phi_n_e.csv",phi_n_e[i])
                np.savetxt(Out_put_dir+"/bag_of_feature_e.txt",total_feat_e_set[i])
                np.savetxt(Out_put_dir+"/bag_of_name_e.txt",total_name_e_set[i])
                np.savetxt(Out_put_dir+"/bag_of_function_e.txt",total_function_e_set[i])
                np.savetxt(Out_put_dir+"/class_count.txt",class_count)
                f=open(Out_put_dir+"/dataset"+repr(i)+"/Parameter.txt","w")
                f.write("max_x_value_of_map: "+repr(MAP_X[i])+
                    "\nMax_y_value_of_map: "+repr(MAP_Y[i])+
                    "\nMin_x_value_of_map: "+repr(map_x[i])+
                    "\nMin_y_value_of_map: "+repr(map_y[i])+
                    "\nNumber_of_place: "+repr(concept_num)+
                    "\nData_num: "+repr(data_num[i])+
                    "\nSliding_data_parameter: "+repr(Slide)+
                    "\nNAME_dim: "+repr(NAME_DIM)+
                    "\nDataset: "+Traing_list[i]+
                    "\nEstimated_placeconcept_num: "+repr(C_num)+
                    "\nImage_feature_dim: "+repr(FEATURE_DIM)+
                    "\nFunction_dim: "+repr(FUNCTION_DIM)+
                    "\nStick_breaking_process_max: "+repr(Stick_large_L)+
                    "\nFeature_diric: "+repr(args.feature)+
                    "\nRegion_num: "+repr(R_set[i])+
                    "\nword_not: "+repr(args.word_not)+
                    "\nfunction_not: "+repr(args.function_not)+
                    "\nword_prob: "+repr(cut_prob)+
                    "\ntest_num: "+repr(test_num)
                    )        
                f.close()

                f=open(Out_put_dir+"/dataset"+repr(i)+"/word_data_class.txt","w")
                for d in range(data_num[i]):
                    w=sum(function_set[i][d])
                    if w>0:
                        f.write("data:"+repr(d)+" C_t:"+repr(C_t_set[i][d])+" r_t:"+repr(r_t_set[i][d])+"\n")
                f.close()
                #========Saving Gaussian distrbution===================
                for j in xrange(region_num):
                    np.savetxt(Out_put_dir+"/dataset"+repr(i)+"/mu/gauss_mu"+repr(j)+".csv",Mu_set[i][j])
                    np.savetxt(Out_put_dir+"/dataset"+repr(i)+"/sigma/gauss_sigma"+repr(j)+".csv",Sigma_set[i][j])
                
                #========Saving Probability of p(n_t|r_t), p(w_t|r_t)===================
                region_to_name_prob=np.array([0.0 for k in range(NAME_DIM)])
                region_to_function_prob=np.array([0.0 for k in range(FUNCTION_DIM)])
                for r in range(region_num):
                    for c in range(concept_num):
                        prob=pi_set[i][c][r]
                        region_to_name_prob +=np.array(phi_n[c]*G_set[i][c]*prob)
                        region_to_function_prob +=np.array(phi_w[c]*G_set[i][c]*prob)
                    np.savetxt(Out_put_dir+"/dataset"+repr(i)+"/region_name/region_name"+repr(r)+".txt",region_to_name_prob)
                    np.savetxt(Out_put_dir+"/dataset"+repr(i)+"/region_function/region_function"+repr(r)+".txt",region_to_function_prob)
            
            #==================================================================
            #====================saving concept parameter================================
            print len(phi_f[0])," ",len(phi_w[0])," ",len(phi_n[0])
            np.savetxt(Out_put_dir+"/phi_f.csv",phi_f)
            np.savetxt(Out_put_dir+"/phi_w.csv",phi_w)
            np.savetxt(Out_put_dir+"/phi_n.csv",phi_n)
            np.savetxt(Out_put_dir+"/bag_of_feature.txt",total_feat_set)
            np.savetxt(Out_put_dir+"/bag_of_name.txt",total_name_set)
            np.savetxt(Out_put_dir+"/bag_of_function.txt",total_function_set)

            f=open(Out_put_dir+"/hyper parameter.txt","w")
            f.write("alpha_f: "+repr(alpha_f)+"\nalpha_w: "+repr(alpha_w)
                +("\nalpha_n: ")+repr(alpha_n)+("\ngamma_0: ")+repr(gamma_0)
                +("\nkappa_0: ")+repr(kappa_0)+("\nnu_0: ")+repr(nu_0)
                #+"\nmu_0: "+repr(mu_0)+"\npsai_0: "+repr(psai_0)+
                +"\ngamma: "+repr(gamma)+"\nbeta: "+repr(beta)
                +"\ninitial sigma: "+repr(sigma_init)+"\nsitck break limit: "+repr(Stick_large_L)
                +"\nimage_increment: "+repr(image_increment)+"\nword_increment: "+repr(word_increment)
                +"alpha_e:"+repr(alpha_e)
                +"\npsai: ["+repr(psai_0[0][0])+"\n"+repr(psai_0[1][0])+"\n"+repr(psai_0[2][0])+"\n"+repr(psai_0[3][0])+"]"
                )
            f.close()

            for i in range(data_set_num):
                mutual.mutual_info(Out_put_dir+"/dataset"+repr(i)+"/")

if __name__ == '__main__':
    gibbs()

