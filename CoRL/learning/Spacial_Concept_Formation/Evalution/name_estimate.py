import numpy as np 
import os
import sys
import math
from numpy.linalg import inv, cholesky
import scipy.stats as ss

description_function_list=np.genfromtxt("/home/eagle/Spacial_Concept_Formation/parameter/description_function.txt", delimiter="\n", dtype='S' )
name_list=np.genfromtxt("/home/eagle/Spacial_Concept_Formation/parameter/place_name.txt", dtype='S' )


def Multi_prob(data,phi):
    phi +=1e-300
    phi_log=np.log(phi)
    prob = data.dot(phi_log.T)
    return prob

def normalize(probs):
    prob_factor = 1.0 / sum(probs)
    return [prob_factor * p for p in probs]

def multi_gaussian(x_t,mu_t,sigma_t):
    x_matrix =np.matrix(x_t)
    mu =np.matrix(mu_t)
    sigma =np.matrix(sigma_t)
    a = np.sqrt(np.linalg.det(sigma))*(2*np.pi)**sigma.ndim
    a=math.log(a)
    #print sigma.I
    b = np.linalg.det(-0.5*((x_matrix-mu)*inv(sigma)*(x_matrix-mu).T))
    
    return math.exp(b-a)

def Description_Function_data_read(file,word_increment):

    function_data=[0 for w in range(len(description_function_list))]
    try:
        
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
            for dictionry in description_function_list:
                if data == dictionry:
                    function_data[w]+=word_increment
        function_data=np.array(function_data)
    except IOError:
        function_data=np.array(function_data)

    return function_data

def Name_data_read(file,word_increment):

    name_data=[0 for w in range(len(name_list))]
        
    data=np.genfromtxt(file, delimiter="\n", dtype='S' )

        #print file
            
    try:
        for d in data:
                #print d
            for w,dictionry in enumerate(name_list):
                if d== dictionry:
                    name_data[w]+=word_increment
                    

                        #print name_data[w]
                        

    except TypeError:
            #print d
        for w,dictionry in enumerate(name_list):
            if data == dictionry:
                name_data[w]+=word_increment
                #print data,sum(name_data)

    name_data=np.array(name_data)
    return name_data

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
result_dir=sys.argv[1]
train_list=sys.argv[2]
Training_list=np.genfromtxt(train_list , delimiter="\t", dtype='S' )
mute=1#int(sys.argv[3])
test_num=int(sys.argv[3])
best=int(sys.argv[4])
out_put="name_estimate_new.txt"

print len(name_list[0])
for e in range(len(Training_list)):
    print "dataset:"+repr(e)
    parameter_dir=result_dir+"/dataset"+repr(e)+"/"
    pi=np.loadtxt(parameter_dir+"/pi.csv")
    G=np.loadtxt(parameter_dir+"/G.txt")
    mutual_info=np.loadtxt(parameter_dir+"/mutual_info.csv")
    region_num=len(pi[0])
    sigma_set=[]
    mu_set=[]
    region_count=np.loadtxt(parameter_dir+"/region_count.txt")
    for i in range(region_num):
        sigma=np.loadtxt(parameter_dir+"/sigma/gauss_sigma"+repr(i)+".csv")
        mu=np.loadtxt(parameter_dir+"/mu/gauss_mu"+repr(i)+".csv")
        sigma_set.append(sigma)

        mu_set.append(mu)

    try:
        phi_n=np.loadtxt(parameter_dir+"/phi_n_e.csv")
        phi_f=np.loadtxt(parameter_dir+"/phi_f_e.csv")
        phi_w=np.loadtxt(parameter_dir+"/phi_w_e.csv")
    except IOError:
        phi_n=np.loadtxt(parameter_dir+"/../phi_n.csv")
        phi_f=np.loadtxt(parameter_dir+"/../phi_f.csv")
        phi_w=np.loadtxt(parameter_dir+"/../phi_w.csv")

    prob_n=normalize(np.sum(phi_n,axis=0))[0] 

    all_point=0.0
    number=0
    f_result=open(parameter_dir+"/"+out_put,'w')

    train_dir=Training_list[e]
    point_name=np.loadtxt(train_dir+"/name_point.txt")
    test_file=train_dir+"/test_num.txt"
    test_data_num=test_data_read(test_file,test_num)
    #print test_data_num

    test_feat_set=[]
    test_pose_set=[]
    test_name_set=[]

    for i in test_data_num:
        test_feat=np.loadtxt(train_dir+"/googlenet_prob/"+repr(i)+".csv")
        test_feat_set.append(test_feat)
        pose=np.loadtxt(train_dir+"/position_data/"+repr(i)+".txt")
        test_pose=[pose[0][0],pose[0][1],pose[1][0],pose[1][1]]
        test_pose_set.append(test_pose)
        
        test_name=Name_data_read(train_dir+"/name/"+repr(i)+".txt",1)
        test_name_set.append(test_name)

    test_feat_set=np.array(test_feat_set)
    test_pose_set=np.array(test_pose_set)
    test_name_set=np.array(test_name_set)

        #print test_name
        #====estimate=======
    gauss_prob_set=np.zeros((region_num,len(test_data_num)),dtype=float)
    for r in range(region_num):
        gauss_prob=ss.multivariate_normal.logpdf(test_pose_set,mu_set[r],sigma_set[r])    
        gauss_prob_set[r] +=gauss_prob

    gauss_prob_set=gauss_prob_set.T
    max_region =np.max(gauss_prob_set,axis=1)
    gauss_prob_set =gauss_prob_set -max_region[:,None]
    gauss_prob_set=np.exp(gauss_prob_set)
    sum_set=np.sum(gauss_prob_set,axis=1)
    gauss_prob_set=gauss_prob_set / sum_set[:,None]


    for i in range(len(test_data_num)):
        class_prob=np.array([0.0 for k in range(len(pi))])
        for c in range(len(pi)):
            class_prob[c] +=Multi_prob(test_feat_set[i],phi_f[c])
            class_prob[c] +=math.log(G[c])

            region_prob=0.0
            for r in range(region_num):
                if region_count[r]>0:
                    region_prob +=pi[c][r]*gauss_prob_set[i][r]
            #print region_prob
            class_prob[c] +=math.log(region_prob)
        max_c =np.argmax(class_prob)
        
        class_prob -=class_prob[max_c]
        
        class_prob=np.exp(class_prob)
        
        class_prob=normalize(class_prob)
        prob1_1=np.array([0.0 for k in range(len(name_list))])
        #prob_0_1=np.array([0.0 for k in range(len(name_list))])
        #prob_1_1=np.array([0.0 for k in range(len(name_list))])
        #prob_0_0=np.array([0.0 for k in range(len(name_list))])
        for n in range(len(name_list)):
            for c in range(len(pi)):
                if mute==0:
                    prob1_1[n]+= class_prob[c]*phi_n[c][n]
                else:
                    prob1_1[n]+= class_prob[c]*phi_n[c][n]*mutual_info[c][n]
                #prob0_1[n]+= (1-class_prob[c])*phi_n[c][n]
                #prob1_0[n]+= class_prob[c]*(1-phi_n[c][n])
                #prob0_0[n]+= (1-class_prob[c])*(1-phi_n[c][n])



        prob1_1=normalize(prob1_1)
        index=np.argsort(prob1_1)[::-1]
        prob1_1=np.sort(prob1_1)[::-1]
        f_result.write("Data: "+repr(i)+"\n")
        #print "Data: "+repr(test_data_num[i])
        point=0.0 
        for rank in range(best):
            f_result.write(repr(rank)+": "+name_list[index[rank]]+" "+repr(prob1_1[rank])+"\n")
            #print repr(rank)+": "+name_list[index[rank]]+" "+repr(prob1_1[rank])+"\n"
            
            if test_name_set[i][index[rank]] >0:
                if point<point_name[index[rank]]:
                    #print point,point_name[index[rank]]
                    point=point_name[index[rank]]
        f_result.write("point: "+repr(point)+"\n")
        #print "point: ",point
        number +=1
        #print number
        all_point +=point

    #print number
    f_result.write("total: "+repr(all_point)+"\n")
    f_result.write("average: "+repr(all_point/number)+"\n")
    print "total: "+repr(all_point)+"\n"
    print "average: ",(all_point/number)
    f_result.close()
