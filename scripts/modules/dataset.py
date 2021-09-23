#!/usr/bin/env python
#coding:utf-8

# Third Party
from scipy.io import mmread
from scipy.io import mmwrite

# Self-made Modules
from __init__ import *
from spconavi_math import *
import converter

convert_func = converter.Converter()

class DataSet():
    
    #Read the map data⇒2-dimension array に格納
    def ReadMap(self, outputfile):
        #outputfolder + trialname + navigation_folder + map.csv
        gridmap = np.loadtxt(outputfile + "map.csv", delimiter=",")
        print ("Read map: " + outputfile + "map.csv")
        return gridmap


    #Read the cost map data⇒2-dimension array に格納
    def ReadCostMap(self, outputfile):
        #outputfolder + trialname + navigation_folder + contmap.csv
        costmap = np.loadtxt(outputfile + "costmap.csv", delimiter=",")
        print ("Read costmap: " + outputfile + "contmap.csv")
        return costmap


    #Read the parameters of learned spatial concepts
    def ReadParameters(self, iteration, sample, filename, trialname):
        #THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
        #r = iteration

        W_index = []
        i = 0
        #Read the text file
        for line in open(filename + "/" + trialname + '_w_index_' + str(iteration) + '_' + str(sample) + '.csv', 'r'): 
            itemList = line[:-1].split(',')
            if(i == 1):
                for j in xrange(len(itemList)):
                    if (itemList[j] != ""):
                        W_index = W_index + [itemList[j]]
            i = i + 1
            
        #####パラメータW, μ, Σ, φ, πを入力する#####
        Mu    = [ np.array([ 0.0, 0.0 ]) for i in xrange(K) ]  #[ np.array([[ 0.0 ],[ 0.0 ]]) for i in xrange(K) ]      #位置分布の平均(x,y)[K]
        Sig   = [ np.array([ [0.0, 0.0],[0.0, 0.0] ]) for i in xrange(K) ]      #位置分布の共分散(2×2-dimension)[K]
        W     = [ [0.0 for j in xrange(len(W_index))] for c in xrange(L) ]  #場所の名前(多項分布: W_index-dimension)[L]
        #theta = [ [0.0 for j in xrange(DimImg)] for c in xrange(L) ] 
        Pi    = [ 0.0 for c in xrange(L)]     #場所概念のindexの多項分布(L-dimension)
        Phi_l = [ [0.0 for i in xrange(K)] for c in xrange(L) ]  #位置分布のindexの多項分布(K-dimension)[L]
        
        i = 0
        ##Mu is read from the file
        for line in open(filename + "/" + trialname + '_Myu_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
            itemList = line[:-1].split(',')
            #Mu[i] = np.array([ float(itemList[0]) - origin[0] , float(itemList[1]) - origin[1] ]) / resolution
            Mu[i] = np.array([ float(itemList[0]) , float(itemList[1]) ])
            i = i + 1
        
        i = 0
        ##Sig is read from the file
        for line in open(filename + "/" + trialname + '_S_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
            itemList = line[:-1].split(',')
            #Sig[i] = np.array([[ float(itemList[0])/ resolution, float(itemList[1]) ], [ float(itemList[2]), float(itemList[3])/ resolution ]]) #/ resolution
            Sig[i] = np.array([[ float(itemList[0]), float(itemList[1]) ], [ float(itemList[2]), float(itemList[3]) ]]) 
            i = i + 1
        
        ##phi is read from the file
        c = 0
        #Read the text file
        for line in open(filename + "/" + trialname + '_phi_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
            itemList = line[:-1].split(',')
            for i in xrange(len(itemList)):
                if itemList[i] != "":
                    Phi_l[c][i] = float(itemList[i])
            c = c + 1
            
        ##Pi is read from the file
        for line in open(filename + "/" + trialname + '_pi_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
            itemList = line[:-1].split(',')
            for i in xrange(len(itemList)):
                if itemList[i] != '':
                    Pi[i] = float(itemList[i])
        
        ##W is read from the file
        c = 0
        #Read the text file
        for line in open(filename + "/" + trialname + '_W_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
            itemList = line[:-1].split(',')
            for i in xrange(len(itemList)):
                if itemList[i] != '':
                    #print c,i,itemList[i]
                    W[c][i] = float(itemList[i])
            c = c + 1

        THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
        return THETA
    
    
    def ReadTrellis(self, outputname, temp):
        print ("ReadTrellis")
        # Save the result to the file 
        trellis = np.load(outputname + "_trellis" + str(temp) + ".npy") #, delimiter=",")
        print ("Read trellis: " + outputname + "_trellis" + str(temp) + ".npy")
        return trellis


    #パス計算のために使用したLookupTable_ProbCtをファイル読み込みする
    def ReadLookupTable(self, outputfile):
        # Read the result from the file
        output = outputfile + "LookupTable_ProbCt.csv"
        LookupTable_ProbCt = np.loadtxt(output, delimiter=",")
        print ("Read LookupTable_ProbCt: " + output)
        return LookupTable_ProbCt    


    #Load the probability cost map used for path calculation
    def ReadCostMapProb(self, outputfile):
        # Read the result from the file
        output = outputfile + "CostMapProb.csv"
        CostMapProb = np.loadtxt(output, delimiter=",")
        print ("Read CostMapProb: " + output)
        return CostMapProb  


    #Load the probability value map used for path calculation
    def ReadProbMap(self, outputfile, speech_num):
        # Read the result from the file
        output = outputfile + "N"+str(N_best)+"G"+str(speech_num) + "_PathWeightMap.csv"
        PathWeightMap = np.loadtxt(output, delimiter=",")
        print ("Read PathWeightMap: " + output)
        return PathWeightMap


    def ReadTransition(self, state_num, outputfile):
        Transition = [[approx_log_zero for j in xrange(state_num)] for i in xrange(state_num)] 
        # Read the result from the file
        output_transition = outputfile + "T"+str(T_horizon) + "_Transition_log.csv"
        #Transition = np.loadtxt(outputfile + "_Transition_log.csv", delimiter=",")
        i = 0
        #Read the text file
        for line in open(output_transition, 'r'):
            itemList = line[:-1].split(',')
            for j in xrange(len(itemList)):
                if itemList[j] != '':
                    Transition[i][j] = float(itemList[j])
            i = i + 1
        print ("Read Transition: " + output_transition)
        return Transition


    def ReadTransition_sparse(self, state_num, outputfile):
        #Transition = [[0 for j in xrange(state_num)] for i in xrange(state_num)] 
        # Read the result from the file
        output_transition = outputfile + "T"+str(T_horizon) + "_Transition_sparse.mtx"
        Transition = mmread(output_transition).tocsr()  #.todense()
        print ("Read Transition: " + output_transition)
        return Transition


    def ReadPath(self, outputname, temp):
        # Read the result file
        output = outputname + "_Path" + str(temp) + ".csv"
        Path = np.loadtxt(output, delimiter=",")
        print ("Read Path: " + output)
        return Path


    #Save the path trajectory
    def ViterbiSavePath(self, X_init, Path, Path_ROS, outputname):
        print ("PathSave")
        if (SAVE_X_init == 1):
            # Save the robot initial position to the file (index)
            np.savetxt(outputname + "_X_init.csv", X_init, delimiter=",")
            # Save the robot initial position to the file (ROS)
            np.savetxt(outputname + "_X_init_ROS.csv", convert_func.Array_index_To_Map_coordinates(X_init), delimiter=",")

        # Save the result to the file (index)
        np.savetxt(outputname + "_Path.csv", Path, delimiter=",")
        # Save the result to the file (ROS)
        np.savetxt(outputname + "_Path_ROS.csv", Path_ROS, delimiter=",")
        print ("Save Path: " + outputname + "_Path.csv and _Path_ROS.csv")
        return outputname + "_Path_ROS.csv"

    
    #Save the path trajectory
    def AstarSavePath(self, X_init, X_goal, Path, Path_ROS, outputname):
        print ("PathSave")
        if (SAVE_X_init == 1):
            # Save the robot initial position to the file (index)
            np.savetxt(outputname + "_X_init.csv", X_init, delimiter=",")
            # Save the robot initial position to the file (ROS)
            np.savetxt(outputname + "_X_init_ROS.csv", convert_func.Array_index_To_Map_coordinates(X_init), delimiter=",")
            # Save robot initial position and goal as file (ROS)
            np.savetxt(outputname + "_X_goal_ROS.csv", convert_func.Array_index_To_Map_coordinates(X_goal), delimiter=",")

        # Save the result to the file (index)
        np.savetxt(outputname + "_Path.csv", Path, delimiter=",")
        # Save the result to the file (ROS)
        np.savetxt(outputname + "_Path_ROS.csv", Path_ROS, delimiter=",")
        print ("Save Path: " + outputname + "_Path.csv and _Path_ROS.csv")
        return outputname + "_Path_ROS.csv"

    
    #Save the path trajectory
    def SavePathTemp(self, X_init, Path_one, temp, outputname, IndexMap_one_NOzero, Bug_removal_savior):
        print ("PathSaveTemp")

        #one-dimension array index を2-dimension array index へ⇒ROSの座標系にする
        Path_2D_index = np.array([ IndexMap_one_NOzero[Path_one[i]] for i in xrange(len(Path_one)) ])
        if ( Bug_removal_savior == 0):
            Path_2D_index_original = Path_2D_index + np.array(X_init) - T_horizon
        else:
            Path_2D_index_original = Path_2D_index
        Path_ROS = convert_func.Array_index_To_Map_coordinates(Path_2D_index_original) #

        #Path = Path_2D_index_original #Path_ROS #必要な方をPathとして返す
        # Save the result to the file (index)
        np.savetxt(outputname + "_Path" + str(temp) + ".csv", Path_2D_index_original, delimiter=",")
        # Save the result to the file (ROS)
        np.savetxt(outputname + "_Path_ROS" + str(temp) + ".csv", Path_ROS, delimiter=",")
        print ("Save Path: " + outputname + "_Path" + str(temp) + ".csv and _Path_ROS" + str(temp) + ".csv")


    def SaveTrellis(self, trellis, outputname, temp):
        print ("SaveTrellis")
        # Save the result to the file 
        np.save(outputname + "_trellis" + str(temp) + ".npy", trellis) #, delimiter=",")
        print ("Save trellis: " + outputname + "_trellis" + str(temp) + ".npy")


    #パス計算のために使用したLookupTable_ProbCtをファイル保存する
    def SaveLookupTable(self, LookupTable_ProbCt, outputfile):
        # Save the result to the file 
        output = outputfile + "LookupTable_ProbCt.csv"
        np.savetxt( output, LookupTable_ProbCt, delimiter=",")
        print ("Save LookupTable_ProbCt: " + output)


    #パス計算のために使用した確率値コストマップをファイル保存する
    def SaveCostMapProb(self, CostMapProb, outputfile):
        # Save the result to the file 
        output = outputfile + "CostMapProb.csv"
        np.savetxt( output, CostMapProb, delimiter=",")
        print ("Save CostMapProb: " + output)


    #Save the probability value map used for path calculation
    def SaveProbMap(self, PathWeightMap, outputfile, speech_num):
        # Save the result to the file 
        output = outputfile + "N"+str(N_best)+"G"+str(speech_num) + "_PathWeightMap.csv"
        np.savetxt( output, PathWeightMap, delimiter=",")
        print ("Save PathWeightMap: " + output)


    def SaveTransition(self, Transition, outputfile):
        # Save the result to the file 
        output_transition = outputfile + "T"+str(T_horizon) + "_Transition_log.csv"
        #np.savetxt(outputfile + "_Transition_log.csv", Transition, delimiter=",")
        f = open( output_transition , "w")
        for i in xrange(len(Transition)):
            for j in xrange(len(Transition[i])):
                f.write(str(Transition[i][j]) + ",")
            f.write('\n')
        f.close()
        print ("Save Transition: " + output_transition)


    def SaveTransition_sparse(self, Transition, outputfile):
        # Save the result to the file (.mtx形式)
        output_transition = outputfile + "T"+str(T_horizon) + "_Transition_sparse"
        mmwrite(output_transition, Transition)

        print ("Save Transition: " + output_transition)


    #Save the log likelihood for each time-step
    def SaveLogLikelihood(self, LogLikelihood,flag,flag2, outputname):
        # Save the result to the file 
        if (flag2 == 0):
            if   (flag == 0):
                output_likelihood = outputname + "_Log_likelihood_step.csv"
            elif (flag == 1):
                output_likelihood = outputname + "_Log_likelihood_sum.csv"
        else:
            if   (flag == 0):
                output_likelihood = outputname + "_Log_likelihood_step" + str(flag2) + ".csv"
            elif (flag == 1):
                output_likelihood = outputname + "_Log_likelihood_sum" + str(flag2) + ".csv"

        np.savetxt( output_likelihood, LogLikelihood, delimiter=",")
        print ("Save LogLikekihood: " + output_likelihood)


    #Save the moving distance of the path
    def SavePathDistance(self, Distance, outputname):
        # Save the result to the file 
        output = outputname + "_Distance.csv"
        np.savetxt( output, np.array([Distance]), delimiter=",")
        print ("Save Distance: " + output)


    #Save the moving distance of the path
    def SavePathDistance_temp(self, Distance,temp, outputname):
        # Save the result to the file 
        output = outputname + "_Distance"+str(temp)+".csv"
        np.savetxt( output, np.array([Distance]), delimiter=",")
        print ("Save Distance: " + output)