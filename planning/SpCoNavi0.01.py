#coding:utf-8

###########################################################
# SpCoNavi: Spatial Concept-based Path-Planning Program (開発中)
# Akira Taniguchi 2018/12/13-2019/1/24
###########################################################

##########---遂行タスク---##########
#テスト実行・デバッグ
#事前計算できるものはできるだけpreprocess.pyで行ってファイル読み込みする形にする
##(単語辞書生成、単語認識結果(N-best)、事前計算可能な確率値、Transition(T_horizonごとに保持)、・・・)
#ViterbiPathの計算でlogを使うので最初からlogですべて計算しておく
#Viterbiの計算処理をTransitionをそのまま使わないように変更する（ムダが多く、メモリ消費・処理時間がかかる要因）
#状態数の削減
#並列処理

##########---作業終了タスク---##########
##文字コードをsjis -> sjisのままにした
##現状、Xtは2次元(x,y)として計算(角度(方向)θは考慮しない)
##配列はlistかnumpy.arrayかを注意
##地図が大きいとメモリを大量に消費する・処理が重くなる恐れがある
##状態遷移確率(動作モデル)は確定モデルで近似計算する
##range() -> xrange()
##numbaのjitで高速化（？）and並列化（？）


###未確認
#ReadParameters
#ReadSpeech
#SpeechRecognition
#WordDictionaryUpdate2
#SavePath
#SaveProbMap
#ReadMap
#ReadCostMap
#pi_2_pi
#Prob_Triangular_distribution_pdf
#Motion_Model_Odometry
#Motion_Model_Odometry_No_theta
#PathPlanner
#ViterbiPath

###確認済み

##########---保留---##########
#状態数削減のための近似手法の実装
#状態遷移確率(動作モデル)を確率モデルで計算する実装
#SendPath
#SendProbMap
#PathDistance
#PostProbXt

##############################################
import os
import sys
import glob
import time
import random
#import re
#import csv
#import collections
import numpy as np
import scipy as sp
#from numpy.random import multinomial #,uniform #,dirichlet
from scipy.stats import multivariate_normal,multinomial #,t,invwishart,rv_discrete
#from numpy.linalg import inv, cholesky
from math import pi as PI
from math import cos,sin,sqrt,exp,log,degrees,radians,atan2 #,gamma,lgamma,fabs,fsum
#from sklearn.cluster import KMeans
#from multiprocessing import Pool
#from multiprocessing import Process
#import multiprocessing
#import rospy
#from std_msgs.msg import String
from __init__ import *
from JuliusNbest_dec import *
from submodules import *
from numba import jit, njit, prange

#マップを読み込む⇒確率値に変換⇒2次元配列に格納
def ReadMap(outputfile):
    #outputfolder + trialname + navigation_folder + map.csv
    gridmap = np.loadtxt(outputfile + "map.csv", delimiter=",")
    print "Read map: " + outputfile + "map.csv"
    return gridmap

#コストマップを読み込む⇒確率値に変換⇒2次元配列に格納
def ReadCostMap(outputfile):
    #outputfolder + trialname + navigation_folder + contmap.csv
    costmap = np.loadtxt(outputfile + "costmap.csv", delimiter=",")
    print "Read costmap: " + outputfile + "contmap.csv"
    return costmap

#場所概念の学習済みパラメータを読み込む
def ReadParameters(particle_num, filename):
    #THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
    r = particle_num
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

    W_index = []
    i = 0
    #テキストファイルを読み込み
    for line in open(filename + 'W_list' + str(r) + '.csv', 'r'): 
        itemList = line[:-1].split(',')
        if(i == 0):
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
    for line in open(filename + 'mu' + str(r) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        Mu[i] = np.array([ float(itemList[0]) , float(itemList[1]) ])
        #Mu[i] = np.array([[ float(itemList[0]) ],[ float(itemList[1]) ]])
        i = i + 1
      
    i = 0
    ##Sigの読み込み
    for line in open(filename + 'sig' + str(r) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        Sig[i] = np.array([[ float(itemList[0]), float(itemList[1]) ], [ float(itemList[2]), float(itemList[3]) ]])
        i = i + 1
      
    ##phiの読み込み
    c = 0
    #テキストファイルを読み込み
    for line in open(filename + 'phi' + str(r) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in xrange(len(itemList)):
            if itemList[i] != "":
              Phi_l[c][i] = float(itemList[i])
        c = c + 1
        
    ##Piの読み込み
    for line in open(filename + 'pi' + str(r) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in xrange(len(itemList)):
          if itemList[i] != '':
            Pi[i] = float(itemList[i])
      
    ##Wの読み込み
    c = 0
    #テキストファイルを読み込み
    for line in open(filename + 'W' + str(r) + '.csv', 'r'):
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

#音声ファイルを読み込み
def ReadSpeech(num):
    # wavファイルを指定
    files = glob.glob(speech_folder_go)
    files.sort()
    speech_file = files[num]
    return speech_file

#音声データを受け取り、音声認識を行う⇒文字列配列を渡す・保存
def SpeechRecognition(speech_file, W_index, step, trialname, outputname):
    ##学習した単語辞書を用いて音声認識し、BoWを得る
    St = RecogNbest( speech_file, step, trialname )
    #print St
    Otb_B = [0 for i in xrange(len(W_index))] #[[] for j in xrange(len(St))]
    for j in xrange(len(St)):
      for i in xrange(5):
              St[j] = St[j].replace("<s>", "")
              St[j] = St[j].replace("</s>", "")
              St[j] = St[j].replace(" <s> ", "")
              St[j] = St[j].replace("<sp>", "")
              St[j] = St[j].replace(" </s>", "")
              St[j] = St[j].replace("  ", " ") 
              St[j] = St[j].replace("\n", "")   
      print j,St[j]
      Otb = St[j].split(" ")

      for j2 in xrange(len(Otb)):
          #print n,j,len(Otb_Samp[r][n])
          for i in xrange(len(W_index)):
            #print W_index[i].decode('sjis'),Otb[j]
            if (W_index[i].decode('sjis') == Otb[j2] ):  #'utf8'
              Otb_B[i] = Otb_B[i] + 1
              #print W_index[i].decode('sjis'),Otb[j]
    print Otb_B

    # 認識結果をファイル保存
    f = open( outputname + "_St.csv" , "w") # , "sjis" )
    for i in xrange(len(St)):
        f.write(St[i].encode('sjis'))
        f.write('\n')
    f.close()

    return Otb_B

#角度を[-π,π]に変換(参考：https://github.com/AtsushiSakai/PythonRobotics)
def pi_2_pi(angle):
    return (angle + PI) % (2 * PI) - PI

#三角分布の確率密度関数
def Prob_Triangular_distribution_pdf(a,b):
    prob = max( 0, ( 1 / (sqrt(6)*b) ) - ( abs(a) / (6*(b**2)) ) )
    return prob

#確率分布の選択
def Motion_Model_Prob(a,b):
    if (MotionModelDist == "Gauss"):
      p = multivariate_normal.pdf(a, 0, b)
    elif (MotionModelDist == "Triangular"):
      p = Prob_Triangular_distribution_pdf(a, b)
    return p

#オドメトリ動作モデル(確率ロボティクスp.122) #現状、不使用
def Motion_Model_Odometry(xt,ut,xt_1):
    #ut = (xt_1_bar, xt_bar), xt_1_bar = (x_bar, y_bar, theta_bar), xt_bar = (x_dash_bar, y_dash_bar, theta_dash_bar)
    x_dash, y_dash, theta_dash = xt
    x, y, theta = xt_1
    xt_1_bar, xt_bar = ut
    x_dash_bar, y_dash_bar, theta_dash_bar = xt_bar
    x_bar, y_bar, theta_bar = xt_1_bar

    delta_rot1  = atan2(y_dash_bar - y_bar, x_dash_bar - x_bar) - theta_bar
    delta_trans = sqrt( (x_dash_bar - x_bar)**2 + (y_dash_bar - y_bar)**2 )
    delta_rot2  = theta_dash_bar - theta_bar - delta_rot1

    delta_rot1_hat  = atan2(y_dash - y, x_dash - x) - theta
    delta_trans_hat = sqrt( (x_dash - x)**2 + (y_dash - y)**2 )
    delta_rot2_hat  = theta_dash - theta - delta_rot1_hat

    p1 = Motion_Model_Prob(pi_2_pi(delta_rot1 - delta_rot1_hat), odom_alpha1*(delta_rot1_hat**2) + odom_alpha2*(delta_trans_hat**2))
    p2 = Motion_Model_Prob(delta_trans - delta_trans_hat, odom_alpha3*(delta_trans_hat**2) + odom_alpha4*(delta_rot1_hat**2+delta_rot2_hat**2))
    p3 = Motion_Model_Prob(pi_2_pi(delta_rot2 - delta_rot2_hat), odom_alpha1*(delta_rot2_hat**2) + odom_alpha2*(delta_trans_hat**2))

    return p1*p2*p3

#オドメトリ動作モデル(簡略版) #角度は考慮せず、移動量に応じて確率が決まる(ドーナツ型分布)
def Motion_Model_Odometry_No_theta(xt,ut,xt_1):
    #ut = (xt_1_bar, xt_bar), xt_1_bar = (x_bar, y_bar), xt_bar = (x_dash_bar, y_dash_bar)
    #utは相対的な位置関係で良い
    x_dash, y_dash = xt
    x, y = xt_1
    #xt_1_bar, xt_bar = ut
    #x_dash_bar, y_dash_bar = xt_bar
    #x_bar, y_bar = xt_1_bar

    #delta_rot1  = 0 #atan2(y_dash_bar - y_bar, x_dash_bar - x_bar) - theta_bar
    delta_trans = cmd_vel #sqrt( (x_dash_bar - x_bar)**2 + (y_dash_bar - y_bar)**2 )
    #delta_rot2  = 0 #theta_dash_bar - theta_bar - delta_rot1

    #delta_rot1_hat  = 0 #atan2(y_dash - y, x_dash - x) - theta
    delta_trans_hat = sqrt( (x_dash - x)**2 + (y_dash - y)**2 )
    #delta_rot2_hat  = 0 #theta_dash - theta - delta_rot1_hat

    #p1 = Motion_Model_Prob(pi_2_pi(delta_rot1 - delta_rot1_hat), odom_alpha1*(delta_rot1_hat**2) + odom_alpha2*(delta_trans_hat**2))
    p2 = Motion_Model_Prob( delta_trans - delta_trans_hat, odom_alpha3*(delta_trans_hat**2) )# + odom_alpha4*(delta_rot1_hat**2+delta_rot2_hat**2))
    #p3 = Motion_Model_Prob(pi_2_pi(delta_rot2 - delta_rot2_hat), odom_alpha1*(delta_rot2_hat**2) + odom_alpha2*(delta_trans_hat**2))

    return p2  #p1*p2*p3

#動作モデル(独自) #角度は考慮せず、移動先位置に応じて確率が決まる(ガウス分布)
def Motion_Model_Original(xt,ut,xt_1):
    xt = np.array(xt)
    ut = np.array(ut)
    xt_1 = np.array(xt_1)
    dist = np.sum((xt-xt_1)**2)
    #p = Motion_Model_Prob( xt - (xt_1+ut), np.diag([odom_alpha3*dist for i in xrange(len(xt))]) )
    px = Motion_Model_Prob( xt[0] - (xt_1[0]+ut[0]), odom_alpha3*dist )
    py = Motion_Model_Prob( xt[1] - (xt_1[1]+ut[1]), odom_alpha3*dist )
    return px*py

#ROSの地図座標系をPython内の2次元配列のインデックス番号に対応付ける
def Map_coordinates_To_Array_index(X):
    X = np.array(X)
    Index = np.round( (X - origin) / resolution ).astype(int) #四捨五入してint型にする
    #Index = np.array([0,0])
    #Index[0] = ( (X[0] - origin[0]) / resolution ).astype(int)
    #Index[1] = ( (X[1] - origin[1]) / resolution ).astype(int)
    return Index

#Python内の2次元配列のインデックス番号からROSの地図座標系への変換
def Array_index_To_Map_coordinates(Index):
    Index = np.array(Index)
    X = np.array( (Index * resolution) + origin )
    return X

#@jit(nopython=True, parallel=True)
@jit #(parallel=True)
def PostProbMap_jit(gridmap,Mu,Sig,Phi_l,LookupTable_ProbCt,map_length,map_width,L,K):
    PostProbMap = np.zeros((map_length,map_width))
    #愚直な実装(for文の多用)
    #memo: np.vectorize or np.frompyfunc の方が処理は早い？    
    for length in prange(map_length):
      for width in prange(map_width):
        if (gridmap[length][width] != -1) and (gridmap[length][width] != 100):  #gridmap[][]が障害物(100)または未探索(-1)であれば計算を省く
          X_temp = Array_index_To_Map_coordinates([width, length])  #地図と縦横の座標系の軸が合っているか要確認
          #print X_temp,Mu
          sum_i_GaussMulti = [ np.sum([multivariate_normal.pdf(X_temp, mean=Mu[k], cov=Sig[k]) * Phi_l[c][k] for k in xrange(K)]) for c in xrange(L) ]
          sum_c_ProbCtsum_i = np.sum( LookupTable_ProbCt * sum_i_GaussMulti )
          PostProbMap[length][width] = sum_c_ProbCtsum_i
    return PostProbMap

"""
#numpyで計算しているところがないため、高速化できているか不明
#@jit(nopython=True, parallel=True)
@njit(parallel=True)
def Transition_jit(state_num,IndexMap_one_NOzero,MoveIndex_list):
    Transition = [[0]*state_num]*state_num #こちらではメモリエラーは起こらなかった
    #今、想定している位置1セルと隣接する8セルのみの遷移を考えるようにすればよい
    for n in prange(state_num):
      Index_2D = IndexMap_one_NOzero[n] #.tolist()
      MoveIndex_list_n = MoveIndex_list + Index_2D #絶対座標系にする
      for c in MoveIndex_list_n.tolist():
        if (c in IndexMap_one_NOzero): #try:
          m = IndexMap_one_NOzero.index(c)  #cは移動可能な状態(セル)とは限らない
          Transition[n][m] = 1 #.0
    return Transition
"""


#@jit(nopython=True, parallel=True)
#@jit #(parallel=True) #なぜかエラーが出る
def Transition_log_jit(state_num,IndexMap_one_NOzero,MoveIndex_list):
    #Transition = np.zeros((state_num,state_num)) * np.log(10**(-300))
    approx_log_zero = np.log(10**(-300)) #ほぼlog(0)の微小値
    Transition = [[approx_log_zero for j in range(state_num)] for i in range(state_num)] 
    print "Memory OK"
    #print IndexMap_one_NOzero
    #今、想定している位置1セルと隣接する8セルのみの遷移を考えるようにすればよい
    for n in prange(state_num):
      Index_2D = IndexMap_one_NOzero[n] #.tolist()
      MoveIndex_list_n = MoveIndex_list + Index_2D #絶対座標系にする
      MoveIndex_list_n_list = MoveIndex_list_n.tolist()
      #print MoveIndex_list_n_list
      ##for m in range(state_num):
      ##  Index_2D_m = IndexMap_one_NOzero[m]   
      ##  if (Index_2D_m in MoveIndex_list_n_list):
      ##    Transition[n][m] = 0.0 #1.0
      ##    #print Index_2D_m, MoveIndex_list_n_list
      ##    #print Index_2D_m in MoveIndex_list_n_list
      for c in prange(len(MoveIndex_list_n_list)):
        #print c
        if (MoveIndex_list_n_list[c] in IndexMap_one_NOzero): #try:
          m = IndexMap_one_NOzero.index(MoveIndex_list_n_list[c])  #cは移動可能な状態(セル)とは限らない
          Transition[n][m] = 0.0 #1 #このインデックスは状態から状態への繊維確率（地図のx,yではない）
        #  print n,m,c
    return Transition

#動的計画法によるグローバルパス推定（SpCoNaviの計算）
def PathPlanner(S_Nbest, X_init, THETA, gridmap, costmap):
    print "[RUN] PathPlanner"
    #THETAを展開
    W, W_index, Mu, Sig, Pi, Phi_l, K, L = THETA

    #MAPの縦横(length and width)のセルの長さを計る
    map_length = len(costmap)
    map_width  = len(costmap[0])
    print "MAP[length][width]:",map_length,map_width

    #事前計算できるものはしておく
    LookupTable_ProbCt = np.array([multinomial.pmf(S_Nbest, sum(S_Nbest), W[c])*Pi[c] for c in xrange(L)])  #Ctごとの確率分布 p(St|W_Ct)×p(Ct|Pi) の確率値
    
    #コストマップを確率の形にする
    CostMapProb = (100.0 - costmap) /100.0

    #ROSの座標系の現在位置を2次元配列のインデックスにする
    X_init_index = X_init ###TEST  #Map_coordinates_To_Array_index(X_init)
    print "Initial Xt:",X_init_index

    print "Please wait for PostProbMap"
    """
    #場所概念部分の重みマップの初期化
    PostProbMap = np.zeros((map_length,map_width))

    #愚直な実装(for文の多用)
    #memo: np.vectorize or np.frompyfunc の方が処理は早い？    
    for length in xrange(map_length):
      for width in xrange(map_width):
        if (gridmap[length][width] != -1) and (gridmap[length][width] != 100):  #gridmap[][]が障害物(100)または未探索(-1)であれば計算を省く
          X_temp = Array_index_To_Map_coordinates([width, length])  #地図と縦横の座標系の軸が合っているか要確認
          #print X_temp,Mu
          sum_i_GaussMulti = [ np.sum([multivariate_normal.pdf(X_temp, mean=Mu[k], cov=Sig[k]) * Phi_l[c][k] for k in xrange(K)]) for c in xrange(L) ]
          sum_c_ProbCtsum_i = np.sum( LookupTable_ProbCt * sum_i_GaussMulti )
          PostProbMap[length][width] = sum_c_ProbCtsum_i
    """

    #####"""
    PostProbMap = PostProbMap_jit(gridmap,Mu,Sig,Phi_l,LookupTable_ProbCt,map_length,map_width,L,K) #マルチCPUで高速化できるかも

    PathWeightMap = CostMapProb * PostProbMap
    print "[Done] PathWeightMap."

    #[TEST]計算結果を先に保存
    SaveProbMap(PathWeightMap, outputname)
    #####"""
    #####PathWeightMap = ReadProbMap(outputname)

    #[メモリ・処理の軽減]初期位置のセルからT_horizonよりも離れた位置のセルをすべて２次元配列から消す([(2*T_horizon)+1][(2*T_horizon)+1]の配列になる)
    if (-T_horizon+X_init_index[0]>=0 and T_horizon+X_init_index[0]<=map_width and -T_horizon+X_init_index[1]>=0 and T_horizon+X_init_index[1]<=map_length):
      PathWeightMap = PathWeightMap[-T_horizon+X_init_index[0]:T_horizon+X_init_index[0]+1, -T_horizon+X_init_index[1]:T_horizon+X_init_index[1]+1] # X[-T+I[0]:T+I[0],-T+I[1]:T+I[1]]
      X_init_index = [T_horizon, T_horizon]
      #再度、MAPの縦横(length and width)のセルの長さを計る
      map_length = len(PathWeightMap)
      map_width  = len(PathWeightMap[0])
    else:
      print "[ERROR] The initial position is outside the map."
      #print X_init, X_init_index


    #計算量削減のため状態数を減らす(状態空間を一次元配列にする⇒0の要素を除く)
    #PathWeight = np.ravel(PathWeightMap)
    PathWeight_one_NOzero = PathWeightMap[PathWeightMap!=0.0]
    state_num = len(PathWeight_one_NOzero)
    print "PathWeight_one_NOzero state_num:", state_num

    #地図の2次元配列インデックスと一次元配列の対応を保持する
    IndexMap = np.array([[(i,j) for j in xrange(map_width)] for i in xrange(map_length)])
    #IndexMap_one = np.ravel(IndexMap)
    IndexMap_one_NOzero = IndexMap[PathWeightMap!=0.0].tolist() #先にリスト型にしてしまう
    print "IndexMap_one_NOzero"


    #1次元配列上の初期位置
    if (X_init_index in IndexMap_one_NOzero): #try:
      X_init_index_one = IndexMap_one_NOzero.index(X_init_index) #.tolist())  
      #np.arrayでは2次元配列のインデックスが取得できなかった
    else:
      #except IndexError:
      print "[ERROR] The initial position is not a movable position on the map."
      #print X_init, X_init_index
      X_init_index_one = 0
    print "Initial index",X_init_index_one

    #移動先候補のインデックス座標のリスト(相対座標)
    MoveIndex_list = MovePosition_2D([0,0]) #.tolist()
    #MoveIndex_list = np.round(MovePosition(X_init_index)).astype(int)
    print "MoveIndex_list"

    #print "len(PathWeight_one_NOzero):"
    #状態遷移確率(動作モデル)の計算
    #TransitionMap = np.array([[0.0 for j in xrange(map_width)] for i in xrange(map_length)])
    #####
    #Transition = 
    #[[0.0]*state_num]*state_num #こちらではメモリエラーは起こらなかった
    #np.array([[0.0 for m in xrange(len(PathWeight_one_NOzero))] for n in xrange(len(PathWeight_one_NOzero))])  ##メモリが大量に消費されてしまう！！！
    #後の処理のためにnumpyにしない(?)
    print "Please wait for Transition"

    """
    #今、想定している位置1セルと隣接する8セルのみの遷移を考えるようにすればよい
    for n in xrange(state_num):
      Index_2D = IndexMap_one_NOzero[n] #.tolist()
      MoveIndex_list_n = MoveIndex_list + Index_2D #絶対座標系にする
      for c in MoveIndex_list_n.tolist():
        if (c in IndexMap_one_NOzero): #try:
          m = IndexMap_one_NOzero.index(c)  #cは移動可能な状態(セル)とは限らない
          Transition[n][m] = 1 #.0
        #except IndexError:
        #  m = 0
        #if (Index_2D.tolist() == m):
    """

    #"""
    #Transition = Transition_jit(state_num,IndexMap_one_NOzero,MoveIndex_list)
    Transition = Transition_log_jit(state_num,IndexMap_one_NOzero,MoveIndex_list)

    #[TEST]計算結果を先に保存
    SaveTransition(Transition, outputname)
    #"""

    #####Transition = [[np.log(10**(-300)) for j in range(state_num)] for i in range(state_num)] 
    #####Transition = ReadTransition(Transition, outputname)


    #IndexMap_one_NOzero内の2次元配列上のインデックスと一致した要素のみ確率1を持つようにする
    #Transition = MovePosition_2D([i,j])

    #Transition_one = np.ravel(Transition)
    Transition_one_NOzero = Transition #[PathWeightMap!=0.0]
    print "[Done] Transition distribution."


    #Viterbi Algorithmを実行
    Path_one = ViterbiPath(X_init_index_one, np.log(PathWeight_one_NOzero), Transition_one_NOzero)

    #1次元配列のインデックスを2次元配列のインデックスへ⇒ROSの座標系にする
    Path_2D_index = [ IndexMap_one_NOzero[Path_one[i]] for i in xrange(len(Path_one)) ]
    Path_2D_index_original = np.array(Path_2D_index) + np.array(X_init) - T_horizon
    Path_ROS = Array_index_To_Map_coordinates(Path_2D_index_original) #

    Path = Path_2D_index_original #Path_ROS #必要な方をPathとして返す
    #ROSのパスの形式にできればなおよい
    print "Init:", X_init
    print "Path:\n", Path
    return Path, PathWeightMap

"""
#あるXtにおける軌道の事後確率(重み)の計算 [状態遷移確率(動作モデル)以外]
def PostProbXt(X, Mu, sig):
    PostProb = 0.0
    print X
    #事前計算できるものはしておく
    PostProb = multivariate_normal.pdf(X, mean=Mu, cov=sig)

    #パスの推定の計算
    for t in xrange(T_horizon):
        print "time:", t
        #if (Dynamics == 1):
        for i in xrange(map_length):
          for j in xrange(map_width):
            if (costmap[i][j] != 0):
              PathWeightMap[i][j] = PathWeightMap[i][j] #* PostProbXt([i,j], S_Nbest, THETA)
        #elif (Dynamics == 0):

    return PostProb
"""

"""
#移動位置の候補を現在の位置(2次元配列のインデックス)とロボットの移動量から計算
def MovePosition(Xt):
    PostPosition_list = []
    for i in xrange(1, 360):
      theta = math.radians(i)
      PostPosition = np.array(Xt) + [np.cos(theta)*cmd_vel, np.sin(theta)*cmd_vel]
      PostPosition_list += [PostPosition]
    return PostPosition_list
"""

#移動位置の候補：現在の位置(2次元配列のインデックス)の近傍8セル+現在位置1セル
def MovePosition_2D(Xt): 
    PostPosition_list = np.array([ [-1,-1],[-1,0],[-1,1], [0,-1],[0,0], [0,1], [1,-1],[1,0],[1,1] ]) + np.array(Xt)
    return PostPosition_list


#Viterbi Path計算用関数(参考：https://qiita.com/kkdd/items/6cbd949d03bc56e33e8e)
def update(cost, trans, emiss):
    COST, INDEX = range(2)  #0,1
    arr = [c[COST]+t for c, t in zip(cost, trans)] #transもlogなら本来はc[COST]+t
    #min_arr = min(arr)
    #return min_arr + emiss, arr.index(min_arr)
    max_arr = max(arr)
    #print max_arr + emiss, arr.index(max_arr)
    return max_arr + emiss, arr.index(max_arr)

#とある状態xtにおける遷移確率0の配列要素は除く?
#def transition(m, n):
#    return [[1.0 for i in xrange(m)] for j in xrange(n)]

#def emission(n):
#    return [random.random() for j in xrange(n)]

#ViterbiPathを計算してPath(軌道)を返す
def ViterbiPath(X_init, PathWeight, Transition):
    #Path = [[0,0] for t in xrange(T_horizon)]  #各tにおけるセル番号[x,y]
    print "Start Viterbi Algorithm"
    #Xt = X_init #自己位置の初期化

    COST, INDEX = range(2)  #0,1
    INITIAL = (np.log(10.0**(-300)), X_init)  # (cost, index) #indexに初期値の一次元配列インデックスを入れる
    #print "Initial:",X_init

    #nstates = [1] + [len(Transition[i]) for i in xrange(T_horizon)] + [1] #[1,2,4,4,2,1] #ステップごとの状態数
    #nstates = [1] + [2,4,4,2,3] + [1] #初期位置は一意に与えられる #最後の遷移確率は一様にすればよいはず
    cost = [INITIAL for i in xrange(len(PathWeight))] # for i in xrange(nstates[0])]
    cost[X_init] = (0.0, X_init)
    trellis = []

    #Forward
    print "Forward"
    for i in prange(T_horizon):  #len(nstates)): #計画区間まで1セルずつ移動していく+1+1
        e = PathWeight #emission(PathWeigh)  #emission(nstates[i])
        m = Transition #transition(nstates[i-1], nstates[i]) #一つ前から現在への遷移
        
        cost = [update(cost, t, f) for t, f in zip(m, e)]
        trellis.append(cost)
        #print "i", i, [(c[COST], c[INDEX]) for c in cost] #前のノードがどこだったか（どこから来たか）を記録している

    e_last = [0.0]
    m_last = [[0.0 for i in range(len(PathWeight))]]
    cost = [update(cost, t, f) for t, f in zip(m_last, e_last)]
    trellis.append(cost)

    #Backward
    print "Backward"
    #last = [trellis[-1][i][0] for i in xrange(len(trellis[-1]))]
    path = [0]  #[last.index(max(last))] #最終的にいらないが計算上必要⇒最後のノードの最大値インデックスを保持
    #print "last",last,"max",path

    for x in reversed(trellis):
        path = [x[path[0]][INDEX]] + path
        #print "x", len(x), x
    path = path[1:len(path)-1]
    print 'Maximum prob path:', path #[1:len(path)-1]
    return path

#推定されたパスを（トピックかサービスで）送る
#def SendPath(Path):

#パスをファイル保存する（形式未定）
def SavePath(X_init, Path, outputname):
    print "PathSave"
    # ロボット初期位置をファイル保存
    np.savetxt(outputname + "_X_init.csv", X_init, delimiter=",")
    #f = open( outputname + "_X_init.csv" , "w")
    #f.write(X_init)
    #f.close()

    # 結果をファイル保存
    np.savetxt(outputname + "_Path.csv", Path, delimiter=",")
    #f = open( outputname + "_Path.csv" , "w")
    #for i in xrange(len(Path)):
    #    f.write(Path[i] + ",")
    #    #f.write('\n')
    #f.close()


#パス計算のために使用した確率値マップを（トピックかサービスで）送る
#def SendProbMap(PathWeightMap):

#パス計算のために使用した確率値マップをファイル保存する
def SaveProbMap(PathWeightMap, outputname):
    #print PathWeightMap
    # 結果をファイル保存
    np.savetxt(outputname + "PathWeightMap.csv", PathWeightMap, delimiter=",")
    print "Save PathWeightMap: " + outputname + "PathWeightMap.csv"

    """
    f = open( outputname + "_PathWeightMap.csv" , "w")# , "sjis" )
    for i in xrange(len(PathWeightMap)):
      for j in xrange(len(PathWeightMap[i])):
        f.write(str(PathWeightMap[i][j]) + ",")
      f.write('\n')
    f.close()
    """

#パス計算のために使用した確率値マップをファイル読み込みする
def ReadProbMap(outputname):
    #print PathWeightMap
    # 結果をファイル読み込み
    PathWeightMap = np.loadtxt(outputname + "PathWeightMap.csv", delimiter=",")
    print "Read PathWeightMap: " + outputname + "PathWeightMap.csv"
    return PathWeightMap

def SaveTransition(Transition, outputname):
    # 結果をファイル保存
    #np.savetxt(outputname + "_Transition_log.csv", Transition, delimiter=",")
    f = open( outputname + "_Transition_log.csv" , "w")
    for i in xrange(len(Transition)):
      for j in xrange(len(Transition[i])):
        f.write(str(Transition[i][j]) + ",")
      f.write('\n')
    f.close()
    print "Save Transition: " + outputname + "_Transition_log.csv"

def ReadTransition(Transition, outputname):
    # 結果をファイル読み込み
    #Transition = np.loadtxt(outputname + "_Transition_log.csv", delimiter=",")
    i = 0
    #テキストファイルを読み込み
    for line in open(outputname + "_Transition_log.csv", 'r'):
        itemList = line[:-1].split(',')
        for j in xrange(len(itemList)):
            if itemList[j] != '':
              #print c,i,itemList[i]
              Transition[i][j] = float(itemList[j])
        i = i + 1

    print "Read Transition: " + outputname + "_Transition_log.csv"
    return Transition

##単語辞書読み込み書き込み追加
def WordDictionaryUpdate2(step, filename, W_list):
  LIST = []
  LIST_plus = []
  i_best = len(W_list)
  hatsuon = [ "" for i in xrange(i_best) ]
  TANGO = []
  ##単語辞書の読み込み
  for line in open('./lang_m/' + lang_init, 'r'):
      itemList = line[:-1].split('	')
      LIST = LIST + [line]
      for j in xrange(len(itemList)):
          itemList[j] = itemList[j].replace("[", "")
          itemList[j] = itemList[j].replace("]", "")
      
      TANGO = TANGO + [[itemList[1],itemList[2]]]
      
  #print TANGO
  if (1):
    ##W_listの単語を順番に処理していく
    for c in xrange(i_best):    # i_best = len(W_list)
          #W_list_sj = unicode(MI_best[c][i], encoding='shift_jis')
          W_list_sj = unicode(W_list[c], encoding='shift_jis')
          if len(W_list_sj) != 1:  ##１文字は除外
            #for moji in xrange(len(W_list_sj)):
            moji = 0
            while (moji < len(W_list_sj)):
              flag_moji = 0
              #print len(W_list_sj),str(W_list_sj),moji,W_list_sj[moji]#,len(unicode(W_list[i], encoding='shift_jis'))
              
              for j in xrange(len(TANGO)):
                if (len(W_list_sj)-2 > moji) and (flag_moji == 0): 
                  #print TANGO[j],j
                  #print moji
                  if (unicode(TANGO[j][0], encoding='shift_jis') == W_list_sj[moji]+"_"+W_list_sj[moji+2]) and (W_list_sj[moji+1] == "_"): 
                    ###print moji,j,TANGO[j][0]
                    hatsuon[c] = hatsuon[c] + TANGO[j][1]
                    moji = moji + 3
                    flag_moji = 1
                    
              for j in xrange(len(TANGO)):
                if (len(W_list_sj)-1 > moji) and (flag_moji == 0): 
                  #print TANGO[j],j
                  #print moji
                  if (unicode(TANGO[j][0], encoding='shift_jis') == W_list_sj[moji]+W_list_sj[moji+1]):
                    ###print moji,j,TANGO[j][0]
                    hatsuon[c] = hatsuon[c] + TANGO[j][1]
                    moji = moji + 2
                    flag_moji = 1
                    
                #print len(W_list_sj),moji
              for j in xrange(len(TANGO)):
                if (len(W_list_sj) > moji) and (flag_moji == 0):
                  #else:
                  if (unicode(TANGO[j][0], encoding='shift_jis') == W_list_sj[moji]):
                      ###print moji,j,TANGO[j][0]
                      hatsuon[c] = hatsuon[c] + TANGO[j][1]
                      moji = moji + 1
                      flag_moji = 1
            print W_list_sj,hatsuon[c]
          else:
            print W_list_sj, "(one name)" #W_list[c]
            
    print JuliusVer,HMMtype
    if (JuliusVer == "v4.4" and HMMtype == "DNN"):
      #hatsuonのすべての単語の音素表記を"*_I"にする
      for i in xrange(len(hatsuon)):
        hatsuon[i] = hatsuon[i].replace("_S","_I")
        hatsuon[i] = hatsuon[i].replace("_B","_I")
        hatsuon[i] = hatsuon[i].replace("_E","_I")
      
      #hatsuonの単語の先頭の音素を"*_B"にする
      for i in xrange(len(hatsuon)):
        #onsohyoki_index = onsohyoki.find(target)
        hatsuon[i] = hatsuon[i].replace("_I","_B", 1)
        
        #hatsuonの単語の最後の音素を"*_E"にする
        hatsuon[i] = hatsuon[i][0:-2] + "E "
        
        #hatsuonの単語の音素の例外処理（N,q）
        hatsuon[i] = hatsuon[i].replace("q_S","q_I")
        hatsuon[i] = hatsuon[i].replace("q_B","q_I")
        hatsuon[i] = hatsuon[i].replace("N_S","N_I")
        #print type(hatsuon),hatsuon,type("N_S"),"N_S"
  
  ##各場所の名前の単語ごとに
  meishi = u'名詞'
  meishi = meishi.encode('shift-jis')
  
  ##単語辞書ファイル生成
  fp = open( filename + '/WDnavi.htkdic', 'w')
  for list in xrange(len(LIST)):
    if (list < 3):
        fp.write(LIST[list])
  #if (UseLM == 1):
  if (1):
    ##新しい単語を追加
    c = 0
    for mi in xrange(i_best):    # i_best = len(W_list)
        if hatsuon[mi] != "":
            if ((W_list[mi] in LIST_plus) == False):  #同一単語を除外
              flag_tango = 0
              for j in xrange(len(TANGO)):
                if(W_list[mi] == TANGO[j][0]):
                  flag_tango = -1
              if flag_tango == 0:
                LIST_plus = LIST_plus + [W_list[mi]]
                
                fp.write(LIST_plus[c] + "+" + meishi +"	[" + LIST_plus[c] + "]	" + hatsuon[mi])
                fp.write('\n')
                c = c+1
  fp.close()


"""
#時間を測る(いらないかも)
def TimeMeasurement(start_iter_time, end_iter_time):
    #time_pp: 音声認識終了時からパスプランニング終了まで（SpCoNavi.pyで完結）
    iteration_time = end_iter_time - start_iter_time
    #ファイル書き込み
    fp = open( datafolder + trialname + "/time_pp.txt", 'a')
    fp.write(str(step)+","+str(iteration_time)+"\n")
    fp.close()
"""

"""
#パスの移動距離を計算、ファイル保存
def PathDistance(PathWeightMap):
    Distance = 0

    return Distance
"""

########################################
if __name__ == '__main__': 
    print "[START] SpCoNavi."
    #学習済みパラメータフォルダ名を要求
    trialname = sys.argv[1]
    #print trialname
    #trialname = raw_input("trialname?(folder) >")

    #読み込むパーティクル番号を要求
    particle_num = sys.argv[2] #0

    #ロボット初期位置の候補番号を要求
    init_position_num = sys.argv[3] #0

    #音声命令のファイル番号を要求   
    speech_num = sys.argv[4] #0

    i = 0
    #重みファイルを読み込み
    for line in open(datafolder + trialname + '/'+ str(step) + '/weights.csv', 'r'):   ##読み込む
        if (i == 0):
          MAX_Samp = int(line)
          i += 1
    #最大尤度のパーティクル番号を保存
    particle_num = MAX_Samp

    #開始時刻を保持
    start_time = time.time()

    ##FullPath of folder
    filename = datafolder + trialname + "/" + str(step) +"/"
    print filename, particle_num
    outputfile = outputfolder + trialname + navigation_folder
    outputname = outputfile + "T"+str(T_horizon)+"N"+str(N_best)+"A"+str(Approx)+"S"+str(init_position_num)+"G"+str(speech_num)

    #Makedir( outputfolder + trialname )
    Makedir( outputfile )
    #Makedir( outputname )

    #学習済みパラメータの読み込み
    THETA = ReadParameters(particle_num, filename)
    #THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
    W_index = THETA[1]
    
    ##単語辞書登録
    WordDictionaryUpdate2(step, filename, W_index)     

    ##マップの読み込み
    gridmap = ReadMap(outputfile)
    ##コストマップの読み込み
    costmap = ReadCostMap(outputfile)

    #音声認識開始時刻(初期化読み込み処理終了時刻)を保持
    start_recog_time = time.time()
    time_init = start_recog_time - start_time
    fp = open( outputname + "time_init.txt", 'w')
    fp.write(str(time_init)+"\n")
    fp.close()

    ##音声ファイルを読み込み
    speech_file = ReadSpeech(int(speech_num))

    #音声認識
    S_Nbest = SpeechRecognition(speech_file, W_index, step, trialname, outputname)

    #音声認識終了時刻（PP開始時刻）を保持
    end_recog_time = time.time()
    time_recog = end_recog_time - start_recog_time
    fp = open( outputname + "time_recog.txt", 'w')
    fp.write(str(time_recog)+"\n")
    fp.close()

    #パスプランニング
    Path, PathWeightMap = PathPlanner(S_Nbest, X_candidates[int(init_position_num)], THETA, gridmap, costmap)

    #PP終了時刻を保持
    end_pp_time = time.time()
    time_pp = end_recog_time - end_pp_time
    fp = open( outputname + "time_pp.txt", 'w')
    fp.write(str(time_pp)+"\n")
    fp.close()

    #パスの移動距離
    #Distance = PathDistance(Path)

    #パスを送る
    #SendPath(Path)
    #パスを保存
    SavePath(X_candidates[int(init_position_num)], Path, outputname)


    #確率値マップを送る
    #SendProbMap(PathWeightMap)
    #確率値マップを保存
    #####SaveProbMap(PathWeightMap, outputname)
    print "[END] SpCoNavi."


########################################

