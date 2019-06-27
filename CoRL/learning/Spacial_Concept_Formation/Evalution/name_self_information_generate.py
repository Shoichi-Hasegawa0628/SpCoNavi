import sys
import numpy as np

train_list=sys.argv[1]
training_list=np.genfromtxt(train_list , delimiter="\t", dtype='S' )
name_list=np.genfromtxt("/home/eagle/Spacial_Concept_Formation/parameter/place_name.txt", dtype='S' )

#print env_para
cut_prob=1.0

def Name_data_read_cut(directory,word_increment,DATA_NUM,test_num):
    name_data_set=[]
    cut=cut_prob
    for i in range(DATA_NUM):
        name_data=[0 for w in range(len(name_list))]
        if  (i in test_num)==False:
            try:
                file=directory+"/name/"+repr(i)+".txt"
                #print file
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


                except IOError:
                    #print d
                    for dictionry in name_list:
                        if d == dictionry:
                            name_data[w]+=increment
                name_data_set.append(name_data)
            except:
                pass
            name_data=np.array(name_data)
            #print name_data
            
        #else:
            #print i,"is test data."
    return name_data_set
sum_name=[0.0 for i in range(len(name_list))]
word_num=0.0
for e in range(len(training_list)):
    env_para=np.genfromtxt(training_list[e]+"/Environment_parameter.txt", dtype='S',delimiter=" ")
    data_num=int(env_para[6][1])-int(env_para[5][1])+1
    name_data=Name_data_read_cut(training_list[e],1.0,data_num,[0])
    sum_name+= np.sum(name_data,axis=0)
    print len(name_data)
    word_num+=len(name_data)
    print sum_name/word_num
    prob=sum_name/word_num

    for i in range(len(prob)):
        if prob[i]==0:
            prob[i]=1
    name_point=-(np.log2((prob)))
    print name_point
    #for e in range(len(training_list)):
    np.savetxt(training_list[e]+"/name_point.txt",name_point)