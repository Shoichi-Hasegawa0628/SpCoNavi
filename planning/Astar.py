#coding:utf-8

###########################################################
# Path-Planning Program by A star algorithm (開発中)
# Akira Taniguchi 2019/06/24-2019/06/30
# Spacial Thanks: Ryo Ozaki
###########################################################

##実行コマンド
#python ./Astar.py trialname mapname iteration sample init_position_num speech_num
#python ./Astar.py 3LDK_01 s1DK_01 1 0 0 0

import sys
import numpy as np
import matplotlib.pyplot as plt
from __init__ import *

def right(pos):
    return (pos[0], pos[1] + 1)

def left(pos):
    return (pos[0], pos[1] - 1)

def up(pos):
    return (pos[0] - 1, pos[1])

def down(pos):
    return (pos[0] + 1, pos[1])

def Manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

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
            for j in xrange(len(itemList)):
              if (itemList[j] != ""):
                W_index = W_index + [itemList[j]]
        i = i + 1
    
    #####パラメータW、μ、Σ、φ、πを入力する#####
    Mu    = [ np.array([ 0.0, 0.0 ]) for i in xrange(K) ]  #[ np.array([[ 0.0 ],[ 0.0 ]]) for i in xrange(K) ]      #位置分布の平均(x,y)[K]
    Sig   = [ np.array([ [0.0, 0.0],[0.0, 0.0] ]) for i in xrange(K) ]      #位置分布の共分散(2×2次元)[K]
    W     = [ [0.0 for j in xrange(len(W_index))] for c in xrange(L) ]  #場所の名前(多項分布：W_index次元)[L]
    #theta = [ [0.0 for j in xrange(DimImg)] for c in xrange(L) ] 
    Pi    = [ 0.0 for c in xrange(L)]     #場所概念のindexの多項分布(L次元)
    Phi_l = [ [0.0 for i in xrange(K)] for c in xrange(L) ]  #位置分布のindexの多項分布(K次元)[L]
      
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
        for i in xrange(len(itemList)):
            if itemList[i] != "":
              Phi_l[c][i] = float(itemList[i])
        c = c + 1
        
    ##Piの読み込み
    for line in open(filename + "/" + trialname + '_pi_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in xrange(len(itemList)):
          if itemList[i] != '':
            Pi[i] = float(itemList[i])
      
    ##Wの読み込み
    c = 0
    #テキストファイルを読み込み
    for line in open(filename + "/" + trialname + '_W_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in xrange(len(itemList)):
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
        for i in xrange(len(itemList)):
            if itemList[i] != '':
              #print c,i,itemList[i]
              theta[c][i] = float(itemList[i])
        c = c + 1
    """

    THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
    return THETA


#################################################
print "[START] A star algorithm."

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

start = Start_Position[int(init_position_num)]#(83,39) #(92,126) #(126,92) #(1, 1)
#goal  = (95,41) #(97,55) #(55,97) #(height-2, width-2)

##FullPath of folder
filename = outputfolder_SIG + trialname #+ "/" 
print filename, iteration, sample
maze_file = filename + navigation_folder + mapname + ".pgm"
#maze_file = "../CoRL/data/1DK_01/navi/s1DK_01.pgm" #"./sankou/sample_maze.txt"

#maze = np.loadtxt(maze_file, dtype=int)
#height, width = maze.shape

##########
#PGMファイルの読み込み
#http://www.not-enough.org/abe/manual/api-aa09/fileio2.html
infile = open(maze_file , 'rb') #sys.argv[1]

for i in range(4): #最初の4行は無視
    d = infile.readline()
    print(d)
    if (i == 2): #3行目を読み込む
        item   = d[:-1].split(' ')
        height = int(ord(item[0]))
        width  = int(ord(item[1]))

maze = np.zeros((height, width))

for h in range(height):
    for w in range(width):
        d = infile.read(1)
        maze[h][w] = int(255 - ord(d))/255

infile.close
##########

action_functions = [right, left, up, down]
cost_of_actions  = [    1,    1,  1,    1]

#学習済みパラメータの読み込み  #THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
THETA = ReadParameters(iteration, sample, filename, trialname)
W_index = THETA[1]

#####場所概念によってゴール地点を推定
Otb_B = [int(W_index[i] == Goal_Word[int(speech_num)]) * N_best for i in xrange(len(W_index))]
print "BoW:",  Otb_B


goal = (0,0)

#####描画
plt.imshow(maze, cmap="binary")
plt.xticks(rotation=90)
#plt.xticks(np.arange(width), np.arange(width))
#plt.yticks(np.arange(height), np.arange(height))
plt.gca().set_aspect('equal')

# スタートとゴールをプロットする
plt.plot(start[1], start[0], "D", color="tab:blue", markersize=1)
plt.plot(goal[1], goal[0], "D", color="tab:green", markersize=1)

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
        if maze[q] == 1:
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
for i in range(len(Path)):
  plt.plot(Path[i][0], Path[i][1], "D", color="tab:red", markersize=1)

print("Total cost using A* algorithm is "+ str(p_cost))
plt.show()
