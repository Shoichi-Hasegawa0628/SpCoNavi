#!/usr/bin/env python
#coding:utf-8

# 標準ライブラリ
import os
import time

# サードパーティー
import matplotlib.pyplot as plt
import rospy
from std_msgs.msg import String

# 自作ライブラリ
from __init__ import *
from spconavi_math import *
import spconavi_read_data
import spconavi_save_data
import spconavi_viterbi_path_calculate
import spconavi_astar_path_calculate

read_data = spconavi_read_data.ReadingData()
save_data = spconavi_save_data.SavingData()
viterbi_path_calculate = spconavi_viterbi_path_calculate.ViterbiPathPlanner()
astar_path_calculate = spconavi_astar_path_calculate.AstarPathPlanner()


if __name__ == '__main__': 
    pub_next_state = rospy.Publisher("/next_state", String, queue_size=10) 
    rospy.init_node('spconavi_planning')
    print "[START] SpCoNavi."

    #Request a folder name for learned parameters.
    #trialname = sys.argv[1]
    trialname = "3LDK_01"
    #trialname = raw_input("trialname?(folder) >")

    #Request iteration value
    #iteration = sys.argv[2] 
    iteration = 1

    #Request sample value
    #sample = sys.argv[3] 
    sample = 0

    #Request the index number of the robot initial position
    #init_position_num = sys.argv[4] 
    init_position_num = 0

    #Request the file number of the speech instruction   
    #speech_num = sys.argv[5]
    #word = rospy.wait_for_message("/place_id", String, timeout=None)
    #print(int(word.data))
    #speech_num = int(word.data) 
    speech_num = 0

    planning_method = 0 # 1はviterbi, 0はaster

    if (SAVE_time == 1):
        #Substitution of start time
        start_time = time.time()

    ##FullPath of folder
    filename = outputfolder_SIG + trialname #+ "/" 
    outputfile = filename + navigation_folder #outputfolder + trialname + navigation_folder

    #Read the files of learned parameters  #THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
    THETA = read_data.ReadParameters(iteration, sample, filename, trialname)
    W_index = THETA[1]


    if planning_method == 1:
        outputname = outputfile + "T"+str(T_horizon)+"N"+str(N_best)+"A"+str(Approx)+"S"+str(init_position_num)+"G"+str(speech_num)
        Makedir( outputfile )

        if (os.path.isfile(outputfile + "CostMapProb.csv") == False):  #すでにファイルがあれば計算しない
            print "If you do not have map.csv, please run commands for cost map acquisition procedure in advance."
            ##Read the map file
            gridmap = read_data.ReadMap(outputfile)
            ##Read the cost map file
            costmap = read_data.ReadCostMap(outputfile)

            #Change the costmap to the probabilistic costmap
            CostMapProb = read_data.CostMapProb_jit(gridmap, costmap)
            #Write the probabilistic cost map file
            save_data.SaveCostMapProb(CostMapProb, outputfile)
        else:
            #Read the probabilistic cost map file
            CostMapProb = read_data.ReadCostMapProb(outputfile)

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


        #Path-Planning
        Path, Path_ROS, PathWeightMap, Path_one = viterbi_path_calculate.viterbi_path_planner(Otb_B, Start_Position[int(init_position_num)], THETA, CostMapProb, outputfile, speech_num, outputname) #gridmap, costmap)

        if (SAVE_time == 1):
            #PP終了時刻を保持
            end_pp_time = time.time()
            time_pp = end_pp_time - start_time #end_recog_time
            fp = open( outputname + "_time_pp.txt", 'w')
            fp.write(str(time_pp)+"\n")
            fp.close()

        #The moving distance of the path
        Distance = viterbi_path_calculate.PathDistance(Path_one)
      
        #Save the moving distance of the path
        save_data.SavePathDistance(Distance, outputname)

        #Save the path
        next_state  = save_data.ViterbiSavePath(Start_Position[int(init_position_num)], Path, Path_ROS, outputname)
      
        #PathWeightMapとPathからlog likelihoodの値を再計算する
        LogLikelihood_step = np.zeros(T_horizon)
        LogLikelihood_sum = np.zeros(T_horizon)
      
        for t in range(T_horizon):
              #print PathWeightMap.shape, Path[t][0], Path[t][1]
              LogLikelihood_step[t] = np.log(PathWeightMap[ Path[t][0] ][ Path[t][1] ])
              if (t == 0):
                  LogLikelihood_sum[t] = LogLikelihood_step[t]
              elif (t >= 1):
                  LogLikelihood_sum[t] = LogLikelihood_sum[t-1] + LogLikelihood_step[t]
      
      
        #すべてのステップにおけるlog likelihoodの値を保存
        save_data.SaveLogLikelihood(LogLikelihood_step,0,0,outputname)
      
        #すべてのステップにおける累積報酬（sum log likelihood）の値を保存
        save_data.SaveLogLikelihood(LogLikelihood_sum,1,0,outputname)



    else:
        start_list = [0, 0] #Start_Position[int(init_position_num)]#(83,39) #(92,126) #(126,92) #(1, 1)
        start_list[0] = Start_Position[0][0]
        start_list[1] = Start_Position[0][1]
        start = (start_list[0], start_list[1])
        print("Start:", start)

        ##FullPath of folder
        outputname = outputfile + "Astar_Approx_expect_"+"N"+str(N_best)+"A"+str(Approx)+"S"+"X"+str(start[1])+"Y"+str(start[0])+"G"+str(speech_num)
        print("OutPutPath: {}".format(outputname))

        Path, goal, PathWeightMap = astar_path_calculate.astar_path_planner(THETA, W_index, outputfile, speech_num, start)

        if (SAVE_time == 1):
        #PP終了時刻を保持
            end_pp_time = time.time()
            time_pp = end_pp_time - start_time #end_recog_time
            fp = open( outputname + "_time_pp.txt", 'w')
            fp.write(str(time_pp)+"\n")
            fp.close()
    
        for i in range(len(Path)):
            plt.plot(Path[i][0], Path[i][1], "s", color="tab:red", markersize=1)


            #The moving distance of the path
            Distance = astar_path_calculate.PathDistance(Path)

            #Save the moving distance of the path
            save_data.SavePathDistance(Distance)

            print("Path distance using A* algorithm is "+ str(Distance))

            #計算上パスのx,yが逆になっているので直す
            Path_inv = [[Path[t][1], Path[t][0]] for t in range(len(Path))]
            Path_inv.reverse()
            #Path_ROS = Path_inv #使わないので暫定的な措置
            Path_ROS = read_data.Array_index_To_Map_coordinates(Path_inv)
            #パスを保存
            next_state = save_data.AstarSavePath(start, [goal[1], goal[0]], Path_inv, Path_ROS, outputname)


            #Read the emission probability file 
            #PathWeightMap = ReadProbMap(outputfile)

            #Save the log-likelihood of the path
            #PathWeightMapとPathからlog likelihoodの値を再計算する
            LogLikelihood_step = np.zeros(T_horizon)
            LogLikelihood_sum = np.zeros(T_horizon)

            for i in range(T_horizon):
                if (i < len(Path)):
                    t = i
                else:
                    t = len(Path) -1
                #print PathWeightMap.shape, Path[t][0], Path[t][1]
                LogLikelihood_step[i] = np.log(PathWeightMap[ Path_inv[t][0] ][ Path_inv[t][1] ])
                if (t == 0):
                    LogLikelihood_sum[i] = LogLikelihood_step[i]
                elif (t >= 1):
                    LogLikelihood_sum[i] = LogLikelihood_sum[i-1] + LogLikelihood_step[i]


            #すべてのステップにおけるlog likelihoodの値を保存
            save_data.SaveLogLikelihood(outputname, LogLikelihood_step,0,0)

            #すべてのステップにおける累積報酬（sum log likelihood）の値を保存
            save_data.SaveLogLikelihood(outputname, LogLikelihood_sum,1,0)

            plt.savefig(outputname + '_Path.png', dpi=300)#, transparent=True
            #plt.savefig(outputfile + "step/" + conditions + '_Path_Weight' +  str(temp).zfill(3) + '.png', dpi=300)#, transparent=True
            plt.savefig(outputname + '_Path.pdf', dpi=300)#, transparent=True
            plt.clf()
            #print("Processing Time :{}".format(time.time() - base_time))

    #rvizとfollower実行につなげる
    #r = rospy.Rate(10) 
    #while not rospy.is_shutdown():
    for i in range (0, 10, 1):
        pub_next_state.publish(next_state)
        time.sleep(1)
    print("End SpCoNavi !")