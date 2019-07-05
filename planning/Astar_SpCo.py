#coding:utf-8

###########################################################
# Path-Planning Program by A star algorithm (ver. SpCo)
# Akira Taniguchi 2019/06/24-2019/07/02
# Spacial Thanks: Ryo Ozaki
###########################################################

##実行コマンド
#python ./Astar_SpCo.py trialname mapname iteration sample init_position_num speech_num
#python ./Astar_SpCo.py 3LDK_01 s1DK_01 1 0 0 0

import sys
import random
import string
import time
import numpy as np
import scipy as sp
from numpy.linalg import inv, cholesky
from scipy.stats import chi2,multivariate_normal,multinomial
from math import pi as PI
from math import cos,sin,sqrt,exp,log,fabs,fsum,degrees,radians,atan2
import matplotlib.pyplot as plt
import collections
from __init__ import *
#from submodules import *

def right(pos):
    return (pos[0], pos[1] + 1)

def left(pos):
    return (pos[0], pos[1] - 1)

def up(pos):
    return (pos[0] - 1, pos[1])

def down(pos):
    return (pos[0] + 1, pos[1])

def stay(pos):
    return (pos[0], pos[1])

def Manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

#マップを読み込む⇒確率値に変換⇒2次元配列に格納
def ReadMap(outputfile):
    #outputfolder + trialname + navigation_folder + map.csv
    gridmap = np.loadtxt(outputfile + "map.csv", delimiter=",")
    print("Read map: " + outputfile + "map.csv")
    return gridmap

#コストマップを読み込む⇒確率値に変換⇒2次元配列に格納
def ReadCostMap(outputfile):
    #outputfolder + trialname + navigation_folder + contmap.csv
    costmap = np.loadtxt(outputfile + "costmap.csv", delimiter=",")
    print("Read costmap: " + outputfile + "contmap.csv")
    return costmap

#パス計算のために使用した確率値コストマップをファイル読み込みする
def ReadCostMapProb(outputfile):
    # 結果をファイル読み込み
    output = outputfile + "CostMapProb.csv"
    CostMapProb = np.loadtxt(output, delimiter=",")
    print("Read CostMapProb: " + output)
    return CostMapProb

#パス計算のために使用した確率値マップをファイル読み込みする
def ReadProbMap(outputfile):
    # 結果をファイル読み込み
    output = outputfile + "N"+str(N_best)+"G"+str(speech_num) + "_PathWeightMap.csv"
    PathWeightMap = np.loadtxt(output, delimiter=",")
    print( "Read PathWeightMap: " + output)
    return PathWeightMap


#パスをファイル保存する（形式未定）
def SavePath(X_init, X_goal, Path, Path_ROS, outputname):
    print("PathSave")
    if (SAVE_X_init == 1):
      # ロボット初期位置とゴールをファイル保存(index)
      np.savetxt(outputname + "_X_init.csv", X_init, delimiter=",")
      np.savetxt(outputname + "_X_goal.csv", X_goal, delimiter=",")
      # ロボット初期位置とゴールをファイル保存(ROS)
      np.savetxt(outputname + "_X_init_ROS.csv", Array_index_To_Map_coordinates(X_init), delimiter=",")
      np.savetxt(outputname + "_X_goal_ROS.csv", Array_index_To_Map_coordinates(X_goal), delimiter=",")

    # 結果をファイル保存(index)
    np.savetxt(outputname + "_Path.csv", Path, delimiter=",")
    # 結果をファイル保存(ROS)
    np.savetxt(outputname + "_Path_ROS.csv", Path_ROS, delimiter=",")
    print("Save Path: " + outputname + "_Path.csv and _Path_ROS.csv")


#各ステップごとのlog likelihoodの値を保存
def SaveLogLikelihood(outputname, LogLikelihood,flag,flag2):
    # 結果をファイル保存
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
    print("Save LogLikekihood: " + output_likelihood)

#ROSの地図座標系をPython内の2次元配列のインデックス番号に対応付ける
def Map_coordinates_To_Array_index(X):
    X = np.array(X)
    Index = np.round( (X - origin) / resolution ).astype(int) #四捨五入してint型にする
    return Index

#Python内の2次元配列のインデックス番号からROSの地図座標系への変換
def Array_index_To_Map_coordinates(Index):
    Index = np.array(Index)
    X = np.array( (Index * resolution) + origin )
    return X

#パスの移動距離を計算する
def PathDistance(Path):
    Distance = len(collections.Counter(Path))
    print("Path Distance is ", Distance)
    return Distance

#パスの移動距離を保存
def SavePathDistance(Distance):
    # 結果をファイル保存
    output = outputname + "_Distance.csv"
    np.savetxt( output, np.array([Distance]), delimiter=",")
    print("Save Distance: " + output)

#パスの移動距離を保存
def SavePathDistance_temp(Distance,temp):
    # 結果をファイル保存
    output = outputname + "_Distance"+str(temp)+".csv"
    np.savetxt( output, np.array([Distance]), delimiter=",")
    print("Save Distance: " + output)

#場所概念の学習済みパラメータを読み込む
def ReadParameters(iteration, sample, filename, trialname):
    #THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
    #r = iteration
    """
    i = 0
    for line in open(filename + 'index' + str(r) + '.csv', 'r'):   ##読み込む
        itemList = line[:-1].split(',')
        #print itemList
        if (i == 0):
          L = len(itemList) -1
        elif (i == 1):
          K = len(itemList) -1
        i += 1
    print "L:",L,"K:",K
    """

    W_index = []
    i = 0
    #テキストファイルを読み込み
    for line in open(filename + "/" + trialname + '_w_index_' + str(iteration) + '_' + str(sample) + '.csv', 'r'): 
        itemList = line[:-1].split(',')
        if(i == 1):
            for j in range(len(itemList)):
              if (itemList[j] != ""):
                W_index = W_index + [itemList[j]]
        i = i + 1
    
    #####パラメータW、μ、Σ、φ、πを入力する#####
    Mu    = [ np.array([ 0.0, 0.0 ]) for i in range(K) ]  #[ np.array([[ 0.0 ],[ 0.0 ]]) for i in range(K) ]      #位置分布の平均(x,y)[K]
    Sig   = [ np.array([ [0.0, 0.0],[0.0, 0.0] ]) for i in range(K) ]      #位置分布の共分散(2×2次元)[K]
    W     = [ [0.0 for j in range(len(W_index))] for c in range(L) ]  #場所の名前(多項分布：W_index次元)[L]
    #theta = [ [0.0 for j in range(DimImg)] for c in range(L) ] 
    Pi    = [ 0.0 for c in range(L)]     #場所概念のindexの多項分布(L次元)
    Phi_l = [ [0.0 for i in range(K)] for c in range(L) ]  #位置分布のindexの多項分布(K次元)[L]
      
    i = 0
    ##Muの読み込み
    for line in open(filename + "/" + trialname + '_Myu_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        #Mu[i] = np.array([ float(itemList[0]) - origin[0] , float(itemList[1]) - origin[1] ]) / resolution
        Mu[i] = np.array([ float(itemList[0]) , float(itemList[1]) ])
        i = i + 1
      
    i = 0
    ##Sigの読み込み
    for line in open(filename + "/" + trialname + '_S_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        #Sig[i] = np.array([[ float(itemList[0])/ resolution, float(itemList[1]) ], [ float(itemList[2]), float(itemList[3])/ resolution ]]) #/ resolution
        Sig[i] = np.array([[ float(itemList[0]), float(itemList[1]) ], [ float(itemList[2]), float(itemList[3]) ]]) 
        i = i + 1
      
    ##phiの読み込み
    c = 0
    #テキストファイルを読み込み
    for line in open(filename + "/" + trialname + '_phi_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in range(len(itemList)):
            if itemList[i] != "":
              Phi_l[c][i] = float(itemList[i])
        c = c + 1
        
    ##Piの読み込み
    for line in open(filename + "/" + trialname + '_pi_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in range(len(itemList)):
          if itemList[i] != '':
            Pi[i] = float(itemList[i])
      
    ##Wの読み込み
    c = 0
    #テキストファイルを読み込み
    for line in open(filename + "/" + trialname + '_W_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in range(len(itemList)):
            if itemList[i] != '':
              #print c,i,itemList[i]
              W[c][i] = float(itemList[i])
        c = c + 1

    """
    ##thetaの読み込み
    c = 0
    #テキストファイルを読み込み
    for line in open(filename + 'theta' + str(r) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in range(len(itemList)):
            if itemList[i] != '':
              #print c,i,itemList[i]
              theta[c][i] = float(itemList[i])
        c = c + 1
    """

    THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
    return THETA


###↓###発話→場所の認識############################################
def Location_from_speech(Otb_B, THETA):
  #THETAを展開
  W, W_index, Myu, S, pi, phi_l, K, L = THETA

  ##全ての位置分布の平均ベクトルを候補とする
  Xp = []
  
  for j in range(K):
    #x1,y1 = np.random.multivariate_normal([Myu[j][0][0],Myu[j][1][0]],S[j],1).T
    #位置分布の平均値と位置分布からサンプリングした99点の１位置分布に対して合計100点をxtの候補とした
    #for i in range(9):    
    #  x1,y1 = np.mean(np.array([ np.random.multivariate_normal([Myu[j][0][0],Myu[j][1][0]],S[j],1).T ]),0)
    #  Xp = Xp + [[x1,y1]]
    #  print x1,y1
    Xp = Xp + [[Myu[j][0],Myu[j][1]]]
    print(Myu[j][0],Myu[j][1])
    
  pox = [0.0 for i in range(len(Xp))]

  ##位置データごとに
  for xdata in range(len(Xp)):      
        ###提案手法による尤度計算####################
        #Ot_index = 0
        
        #for otb in range(len(W_index)):
        #Otb_B = [0 for j in range(len(W_index))]
        #Otb_B[Ot_index] = 1
        temp = [0.0 for c in range(L)]
        #print Otb_B
        for c in range(L) :
            ##場所の名前、多項分布の計算
            #W_temp = multinomial(W[c])
            #temp[c] = W_temp.pmf(Otb_B)
            temp[c] = multinomial.pmf(Otb_B, sum(Otb_B), W[c]) * pi[c]
            #temp[c] = W[c][otb]
            ##場所概念の多項分布、piの計算
            #temp[c] = temp[c]
            
            ##itでサメーション
            it_sum = 0.0
            for it in range(K):
                """
                if (S[it][0][0] < pow(10,-100)) or (S[it][1][1] < pow(10,-100)) :    ##共分散の値が0だとゼロワリになるので回避
                    if int(Xp[xdata][0]) == int(Myu[it][0]) and int(Xp[xdata][1]) == int(Myu[it][1]) :  ##他の方法の方が良いかも
                        g2 = 1.0
                        print "gauss 1"
                    else : 
                        g2 = 0.0
                        print "gauss 0"
                else : 
                    g2 = gaussian2d(Xp[xdata][0],Xp[xdata][1],Myu[it][0],Myu[it][1],S[it])  #2次元ガウス分布を計算
                """
                g2 = multivariate_normal.pdf(Xp[xdata], mean=Myu[it], cov=S[it])
                it_sum = it_sum + g2 * phi_l[c][it]
                
            temp[c] = temp[c] * it_sum
        
        pox[xdata] = sum(temp)
        
        #print Ot_index,pox[Ot_index]
        #Ot_index = Ot_index + 1
        #POX = POX + [pox.index(max(pox))]
        
        #print pox.index(max(pox))
        #print W_index_p[pox.index(max(pox))]
        
    
  Xt_max = Map_coordinates_To_Array_index( [ Xp[pox.index(max(pox))][0], Xp[pox.index(max(pox))][1] ] ) #[0.0,0.0] ##確率最大の座標候補
  Xt_max_tuple =(Xt_max[1], Xt_max[0])
  print("Goal:", Xt_max_tuple)
  return Xt_max_tuple
###↑###発話→場所の認識############################################



#################################################
print("[START] A star algorithm.")

#地図データの入った部屋環境フォルダ名（学習済みパラメータフォルダ名）を要求
trialname = sys.argv[1]

#マップファイル名を要求
mapname = sys.argv[2]

#iterationを要求
iteration = sys.argv[3] #1

#sampleを要求
sample = sys.argv[4] #0

#ロボット初期位置の候補番号を要求
init_position_num = sys.argv[5] #0

#音声命令のファイル番号を要求   
speech_num = sys.argv[6] #0

if (SAVE_time == 1):
    #開始時刻を保持
    start_time = time.time()

start_list = [0, 0] #Start_Position[int(init_position_num)]#(83,39) #(92,126) #(126,92) #(1, 1)
start_list[0] = int(sys.argv[7]) #0
start_list[1] = int(sys.argv[8]) #0
start = (start_list[0], start_list[1])
print("Start:", start)
#goal  = (95,41) #(97,55) #(55,97) #(height-2, width-2)

##FullPath of folder
filename = outputfolder_SIG + trialname #+ "/" 
print(filename, iteration, sample)
outputfile = filename + navigation_folder #outputfolder + trialname + navigation_folder
#outputname = outputfile + "Astar_SpCo_"+"N"+str(N_best)+"A"+str(Approx)+"S"+str(init_position_num)+"G"+str(speech_num)
outputname = outputfile + "Astar_SpCo_"+"N"+str(N_best)+"A"+str(Approx)+"S"+str(start)+"G"+str(speech_num)

#"T"+str(T_horizon)+"N"+str(N_best)+"A"+str(Approx)+"S"+str(init_position_num)+"G"+str(speech_num)
maze_file = outputfile + mapname + ".pgm"
#maze_file = "../CoRL/data/1DK_01/navi/s1DK_01.pgm" #"./sankou/sample_maze.txt"

#maze = np.loadtxt(maze_file, dtype=int)
#height, width = maze.shape
"""
##########
#PGMファイルの読み込み
#http://www.not-enough.org/abe/manual/api-aa09/fileio2.html
infile = open(maze_file , 'rb') #sys.argv[1]

for i in range(4): #最初の4行は無視
    d = infile.readline()
    print(d[:-1])
    if (i == 2): #3行目を読み込む
        item   = str(d[:-1]).split(' ')
        #print(item)
        height = int((item[0][2:]))
        width  = int((item[1][0:-1]))

maze = np.zeros((height, width))
print(height, width)

for h in range(height):
    for w in range(width):
        d = infile.read(1)
        maze[h][w] = int(255 - ord(d))/255

infile.close
##########
"""
maze = ReadMap(outputfile)
height, width = maze.shape

action_functions = [right, left, up, down, stay] #, migiue, hidariue, migisita, hidarisita]
cost_of_actions  = [    1,    1,  1,    1,    1] #, ,    1,        1,        1,          1]

#学習済みパラメータの読み込み  #THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
THETA = ReadParameters(iteration, sample, filename, trialname)
W_index = THETA[1]

#####場所概念によってゴール地点を推定
Otb_B = [int(W_index[i] == Goal_Word[int(speech_num)]) * N_best for i in range(len(W_index))]
print("BoW:", Otb_B)

#パスプランニング
#Path, Path_ROS, PathWeightMap, Path_one = PathPlanner(Otb_B, Start_Position[int(init_position_num)], THETA, CostMapProb) #gridmap, costmap)

goal = Location_from_speech(Otb_B, THETA) #(0,0)

if (maze[goal[0]][goal[1]] != 0):
    print("[ERROR] goal",maze[goal[0]][goal[1]],"is not 0.")


#####描画
#plt.imshow(maze, cmap="binary")
gridmap = maze
plt.imshow(gridmap + (50+1)*(gridmap == -1), origin='lower', cmap='binary', vmin = 0, vmax = 100, interpolation='none') #, vmin = 0.0, vmax = 1.0)
     
plt.xticks(rotation=90)
plt.tick_params(axis='x', which='major', labelsize=8)
plt.tick_params(axis='y', which='major', labelsize=8)
#plt.xlim([380,800])             #x軸の範囲
#plt.ylim([180,510])             #y軸の範囲
plt.xlabel('X', fontsize=10)
plt.ylabel('Y', fontsize=10)
#plt.xticks(np.arange(width), np.arange(width))
#plt.yticks(np.arange(height), np.arange(height))
plt.gca().set_aspect('equal')

# スタートとゴールをプロットする
plt.plot(start[1], start[0], "D", color="tab:blue", markersize=1)
plt.plot(goal[1], goal[0], "D", color="tab:pink", markersize=1)

#plt.show()

open_list = []
open_list_cost = []
open_list_key = []
closed_list = []
closed_list_cost = []
closed_list_key = []
open_list.append(start)
open_list_cost.append(0)
open_list_key.append(0 + Manhattan_distance(start, goal))
OYA = {}
ko = (0), (0)
Path = []

while open_list:
    sorted_idx = np.argsort(open_list_key, kind="stable")
    pop_idx = sorted_idx[0]
    p = open_list.pop(pop_idx)
    p_cost = open_list_cost.pop(pop_idx)
    p_key = open_list_key.pop(pop_idx)
    closed_list.append(p)
    closed_list_cost.append(p_cost)
    closed_list_key.append(p_key)
    if p == goal:
        break
    for act_func, act_cost in zip(action_functions, cost_of_actions):
        q = act_func(p)
        if (int(maze[q]) != 0):
            continue
        q_cost = p_cost + act_cost
        q_pev = Manhattan_distance(q, goal)
        q_key = q_cost + q_pev

        if q in open_list:
            idx = open_list.index(q)
            key = open_list_key[idx]
            if key > q_key:
                open_list_key[idx] = q_key
                open_list_cost[idx] = q_cost
        elif q in closed_list:
            idx = closed_list.index(q)
            key = closed_list_key[idx]
            if key > q_key:
                closed_list.pop(idx)
                closed_list_cost.pop(idx)
                closed_list_key.pop(idx)
                open_list.append(q)
                open_list_cost.append(q_cost)
                open_list_key.append(q_key)
                #plt.quiver(p[1], p[0], (q[1]-p[1]), (q[0]-p[0]), angles='xy', scale_units='xy', scale=1, color="tab:red")
                OYA[(q[1], q[0])] = (p[1], p[0])
                ko = (q[1]), (q[0])
                #print(ko)
        else:
            open_list.append(q)
            open_list_cost.append(q_cost)
            open_list_key.append(q_key)
            #plt.quiver(p[1], p[0], (q[1]-p[1]), (q[0]-p[0]), angles='xy', scale_units='xy', scale=1, color="tab:red")
            OYA[(q[1], q[0])] = (p[1], p[0])
            ko = (q[1]), (q[0])
            #print(ko)

#最適経路の決定：ゴールから親ノード（どこから来たか）を順次たどっていく
#i = len(OYA)
#for oyako in reversed(OYA):
print(ko,goal)
for i in range(p_cost-1):
  #print(OYA[ko])
  Path = Path + [OYA[ko]]
  ko = OYA[ko]
  #i -= 1

if (SAVE_time == 1):
    #PP終了時刻を保持
    end_pp_time = time.time()
    time_pp = end_pp_time - start_time #end_recog_time
    fp = open( outputname + "_time_pp.txt", 'w')
    fp.write(str(time_pp)+"\n")
    fp.close()
    
for i in range(len(Path)):
  plt.plot(Path[i][0], Path[i][1], "s", color="tab:red", markersize=1)

print("Total cost using A* algorithm is "+ str(p_cost))

#パスの移動距離
Distance = p_cost #PathDistance(Path_one)

#パスの移動距離を保存
SavePathDistance(Distance)

#計算上パスのx,yが逆になっているので直す
Path_inv = [[Path[t][1], Path[t][0]] for t in range(len(Path))]
Path_inv.reverse()
Path_ROS = Path_inv #使わないので暫定的な措置
#パスを保存
SavePath(start, [goal[1], goal[0]], Path_inv, Path_ROS, outputname)


#出力確率のマップの読み込み
PathWeightMap = ReadProbMap(outputfile)

#パスの対数尤度を保存する
#PathWeightMapとPathからlog likelihoodの値を再計算する
LogLikelihood_step = np.zeros(T_horizon)
LogLikelihood_sum = np.zeros(T_horizon)

for i in range(T_horizon):
    if (i < len(Path)):
        t = i
    else:
        t = len(Path) -1
    #print PathWeightMap.shape, Path[t][0], Path[t][1]
    LogLikelihood_step[t] = np.log(PathWeightMap[ Path_inv[t][0] ][ Path[t][1] ])
    if (t == 0):
        LogLikelihood_sum[t] = LogLikelihood_step[t]
    elif (t >= 1):
        LogLikelihood_sum[t] = LogLikelihood_sum[t-1] + LogLikelihood_step[t]


#すべてのステップにおけるlog likelihoodの値を保存
SaveLogLikelihood(outputname, LogLikelihood_step,0,0)

#すべてのステップにおける累積報酬（sum log likelihood）の値を保存
SaveLogLikelihood(outputname, LogLikelihood_sum,1,0)



#plt.show()

#地図をカラー画像として保存
#output = outputfile + "N"+str(N_best)+"G"+str(speech_num)
plt.savefig(outputname + '_Path.png', dpi=300)#, transparent=True
#plt.savefig(outputfile + "step/" + conditions + '_Path_Weight' +  str(temp).zfill(3) + '.png', dpi=300)#, transparent=True
plt.savefig(outputname + '_Path.pdf', dpi=300)#, transparent=True
plt.clf()

