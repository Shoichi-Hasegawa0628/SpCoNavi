#!/usr/bin/env python
#coding:utf-8
import time
base_time = time.time()
import os
import collections
import spconavi_read_data
import spconavi_save_data
from scipy.stats import multinomial
from __init__ import *
from spconavi_math import *
from itertools import izip


read_data = spconavi_read_data.ReadingData()
save_data = spconavi_save_data.SavingData()
############################################################################

trialname = "3LDK_01"
iteration = 1
sample = 0
init_position_num = 0
speech_num = 3 #0, 1, 2, 3


##FullPath of folder
filename = outputfolder_SIG + trialname #+ "/" 
print (filename, iteration, sample)
outputfile = filename + navigation_folder #outputfolder + trialname + navigation_folder
outputname = outputfile + "T"+str(T_horizon)+"N"+str(N_best)+"A"+str(Approx)+"S"+str(init_position_num)+"G"+str(speech_num)


gridmap = read_data.ReadMap(outputfile)
##Read the cost map file
costmap = read_data.ReadCostMap(outputfile)
#Change the costmap to the probabilistic costmap
CostMapProb = read_data.CostMapProb_jit(gridmap, costmap)


THETA = read_data.ReadParameters(iteration, sample, filename, trialname)
W_index = THETA[1]


##Read the speech file
#speech_file = ReadSpeech(int(speech_num))
BoW = [Goal_Word[int(speech_num)]]
if ( "AND" in BoW ):
    BoW = Example_AND
elif ( "OR" in BoW ):
    BoW = Example_OR

Otb_B = [int(W_index[i] in BoW) * N_best for i in xrange(len(W_index))]
print "BoW:",  Otb_B

while (sum(Otb_B) == 0):
    print("[ERROR] BoW is all zero.", W_index)
    word_temp = raw_input("Please word?>")
    Otb_B = [int(W_index[i] == word_temp) * N_best for i in xrange(len(W_index))]
    print("BoW (NEW):",  Otb_B)

S_Nbest = Otb_B


#THETAを展開
W, W_index, Mu, Sig, Pi, Phi_l, K, L = THETA
#length and width of the MAP cells
map_length = len(CostMapProb)     #len(costmap)
map_width  = len(CostMapProb[0])  #len(costmap[0])
print "MAP[length][width]:",map_length,map_width

#Pre-calculation できるものはしておく
LookupTable_ProbCt = np.array([multinomial.pmf(S_Nbest, sum(S_Nbest), W[c])*Pi[c] for c in xrange(L)])  #Ctごとの確率分布 p(St|W_Ct)×p(Ct|Pi) の確率値
###SaveLookupTable(LookupTable_ProbCt, outputfile)
###LookupTable_ProbCt = ReadLookupTable(outputfile)  #Read the result from the Pre-calculation file(計算する場合と大差ないかも)

print "Please wait for PostProbMap"
output = outputfile + "N"+str(N_best)+"G"+str(speech_num) + "_PathWeightMap.csv"
#if (os.path.isfile(output) == False) or (UPDATE_PostProbMap == 1):  #すでにファイルがあれば作成しない
#PathWeightMap = PostProbMap_jit(CostMapProb,Mu,Sig,Phi_l,LookupTable_ProbCt,map_length,map_width,L,K) #マルチCPUで高速化できるかも #CostMapProb * PostProbMap #後の処理のために, この時点ではlogにしない
PathWeightMap = read_data.PostProbMap_nparray_jit(CostMapProb,Mu,Sig,Phi_l,LookupTable_ProbCt,map_length,map_width,L,K) #,IndexMap) 

#[TEST]計算結果を先に保存
save_data.SaveProbMap(PathWeightMap, outputfile, speech_num)
print("Processing Time :{}".format(time.time() - base_time))
print "[Done] PathWeightMap."