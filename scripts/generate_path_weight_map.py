#!/usr/bin/env python
#coding:utf-8

# Standard Library
import time
from itertools import izip

# Third Party
from scipy.stats import multinomial

# Self-made Modules
from __init__ import *
from modules.spconavi_math import *
from modules import dataset, converter

convert_func = converter.Converter()
dataset_func = dataset.DataSet()

trialname = "3LDK_01"
iteration = 1
sample = 0
init_position_num = 0
speech_num = 3 #0, 1, 2, 3

class GeneratePathWeightMap():
    def __init__(self):
        pass

    def calculate_path_weight_map(self):
        ##FullPath of folder
        filename = outputfolder_SIG + trialname #+ "/"
        print (filename, iteration, sample)
        outputfile = filename + navigation_folder #outputfolder + trialname + navigation_folder
        outputname = outputfile + "T"+str(T_horizon)+"N"+str(N_best)+"A"+str(Approx)+"S"+str(init_position_num)+"G"+str(speech_num)


        gridmap = dataset_func.ReadMap(outputfile)
        ##Read the cost map file
        costmap = dataset_func.ReadCostMap(outputfile)
        #Change the costmap to the probabilistic costmap
        CostMapProb = convert_func.CostMapProb_jit(gridmap, costmap)


        THETA = dataset_func.ReadParameters(iteration, sample, filename, trialname)
        W_index = THETA[1]


        ##Read the speech file
        #speech_file = ReadSpeech(int(speech_num))
        BoW = [Goal_Word[int(speech_num)]]
        if ( "AND" in BoW ):
            BoW = Example_AND
        elif ( "OR" in BoW ):
            BoW = Example_OR

        Otb_B = [int(W_index[i] in BoW) * N_best for i in xrange(len(W_index))]
        print ("BoW:",  Otb_B)

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
        print ("MAP[length][width]:",map_length,map_width)

        #Pre-calculation できるものはしておく
        LookupTable_ProbCt = np.array([multinomial.pmf(S_Nbest, sum(S_Nbest), W[c])*Pi[c] for c in xrange(L)])  #Ctごとの確率分布 p(St|W_Ct)×p(Ct|Pi) の確率値
        ###SaveLookupTable(LookupTable_ProbCt, outputfile)
        ###LookupTable_ProbCt = ReadLookupTable(outputfile)  #Read the result from the Pre-calculation file(計算する場合と大差ないかも)

        print ("Please wait for PostProbMap")
        output = outputfile + "N"+str(N_best)+"G"+str(speech_num) + "_PathWeightMap.csv"
        #if (os.path.isfile(output) == False) or (UPDATE_PostProbMap == 1):  #すでにファイルがあれば作成しない
        #PathWeightMap = PostProbMap_jit(CostMapProb,Mu,Sig,Phi_l,LookupTable_ProbCt,map_length,map_width,L,K) #マルチCPUで高速化できるかも #CostMapProb * PostProbMap #後の処理のために, この時点ではlogにしない
        PathWeightMap = convert_func.PostProbMap_nparray_jit(CostMapProb,Mu,Sig,Phi_l,LookupTable_ProbCt,map_length,map_width,L,K) #,IndexMap)

        #[TEST]計算結果を先に保存
        dataset_func.SaveProbMap(PathWeightMap, outputfile, speech_num)
        print ("[Done] PathWeightMap.")

if __name__ == '__main__':
    print ("Ctrl-C is the end of process.")
    rospy.init_node('CostMap', anonymous=True)
    calculate_path_weight = GeneratePathWeightMap()
    calculate_path_weight.calculate_path_weight_map()
    rospy.spin()