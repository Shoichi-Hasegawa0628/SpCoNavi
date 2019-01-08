#coding:utf-8

##############################################
# SpCoNavi Path Planning program (作成中)
# Akira Taniguchi 2018/12/13-2018/12/17
##############################################

##########---遂行タスク---##########
#文字コードをsjis -> sjisのままにした
#ReadCostMap
#PathPlanner
#SavePath
#SaveProbMap
#WordDictionaryUpdate2

##########---作業終了タスク---##########
###未確認
#ReadParameters
#ReadSpeech
#SpeechRecognition


###確認済み

##########---保留---##########
#SendPath
#SendProbMap


##############################################
import os
#import re
import glob
import random
#import csv
#import collections
import numpy as np
import scipy as sp
#from numpy.random import multinomial #,uniform #,dirichlet
from scipy.stats import multivariate_normal,multinomial #,t,invwishart,rv_discrete
from numpy.linalg import inv, cholesky
from math import pi as PI
from math import cos,sin,sqrt,exp,log,fabs,fsum,degrees,radians,atan2,gamma,lgamma
#from sklearn.cluster import KMeans
#from multiprocessing import Pool
#from multiprocessing import Process
#import multiprocessing
from __init__ import *
from submodules import *

#コストマップを読み込む⇒2次元配列に格納(?)
def ReadCostMap():
    #outputfolder + trialname + navigation_folder + contmap.csv
    costmap = np.array([])
    return costmap

#場所概念の学習済みパラメータを読み込む
def ReadParameters(particle_num, filename):
    #THETA = [W,W_index,Myu,Sig,pi,phi_l,K,L]
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
            for j in range(len(itemList)):
              if (itemList[j] != ""):
                W_index = W_index + [itemList[j]]
        i = i + 1
    
    #####パラメータW、μ、Σ、φ、πを入力する#####
    Myu   = [ np.array([[ 0.0 ],[ 0.0 ]]) for i in xrange(K) ]      #位置分布の平均(x,y)[K]
    Sig   = [ np.array([ [0.0, 0.0],[0.0, 0.0] ]) for i in xrange(K) ]      #位置分布の共分散(2×2次元)[K]
    W     = [ [0.0 for j in xrange(len(W_index))] for c in xrange(L) ]  #場所の名前(多項分布：W_index次元)[L]
    #theta = [ [0.0 for j in xrange(DimImg)] for c in xrange(L) ] 
    pi    = [ 0.0 for c in xrange(L)]     #場所概念のindexの多項分布(L次元)
    phi_l = [ [0.0 for i in xrange(K)] for c in xrange(L) ]  #位置分布のindexの多項分布(K次元)[L]
      
    i = 0
    ##Myuの読み込み
    for line in open(filename + 'mu' + str(r) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        Myu[i] = np.array([[ float(itemList[0]) ],[ float(itemList[1]) ]])
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
              phi_l[c][i] = float(itemList[i])
        c = c + 1
        
    ##piの読み込み
    for line in open(filename + 'pi' + str(r) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in xrange(len(itemList)):
          if itemList[i] != '':
            pi[i] = float(itemList[i])
      
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

    THETA = [W,W_index,Myu,Sig,pi,phi_l,K,L]
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
    Otb_B = [0 for i in xrange(len(W_index))] #[[] for j in range(len(St))]
    for j in range(len(St)):
      for i in range(5):
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
            if (W_index[i].decode('sjis') == Otb[j2] ):
            #####if (W_index[i].decode('utf8') == Otb[j] ):
              Otb_B[i] = Otb_B[i] + 1
              #print W_index[i].decode('sjis'),Otb[j]
    print Otb_B
    #S_Nbest = Otb_B

    # 認識結果をファイル保存
    f = open( outputname + "_St.csv" , "w")# , "sjis" )
    for i in range(len(St)):
        #f.write(wordDic[i].encode('sjis'))
        f.write(St[i].encode('sjis'))
        f.write('\n')
    f.close()

    return Otb_B #S_Nbest

#動的計画法によるグローバルパス推定（SpCoNaviの計算）
def PathPlanner(S_Nbest, X_init, THETA, costmap):
    print "S_Nbest: ", S_Nbest

    #THETAを展開
    W = THETA[0]
    W_index = THETA[1]
    Myu = THETA[2]
    Sig = THETA[3]
    pi = THETA[4]
    phi_l = THETA[5]
    K = THETA[6]
    L = THETA[7]

    Xt = X_init #自己位置の初期化

    #MAPの縦横(length and width)のセルの長さを計る
    map_length = len(costmap)
    map_width  = len(costmap[0])

    #事前計算できるものはしておく
    LookupTable_ProbCt = np.array([multinomial.pmf(S_Nbest, sum(S_Nbest), W[c])*pi[c] for c in range(L)])  #Ctごとの確率分布 p(St|W_Ct)×p(Ct|pi) の確率値


    PathWeightMap = costmap 
    #* [ [PostProbXt([i,j], THETA) for j in range(map_width)] for i in range(map_length) ]
    #np.frompyfunc(PostProbXt, [i,j], S_Nbest, THETA)(costmap)
    #[ [costmap[i][j] for j in range(map_width)] for i in range(map_length) ]

    #memo: np.vectorize or np.frompyfunc の方が処理は早い？

    #パスの推定の計算
    for t in range(T_horizon):
        print "time:", t
        #if (Dynamics == 1):
        for i in range(map_length):
          for j in range(map_width):
            if (costmap[i][j] != 0):
              PathWeightMap[i][j] = PathWeightMap[i][j] #* PostProbXt([i,j], S_Nbest, THETA)

        #elif (Dynamics == 0):
        

    Path = ViterbiPath(PathWeightMap)

    return Path, PathWeightMap

#あるXtにおける軌道の事後確率(重み)の計算 [状態遷移確率(動作モデル)以外]
def PostProbXt(Xt, THETA):
    PostProb = 0.0
    #事前計算できるものはしておく


    return PostProb

#ViterbiPathを計算してPath(軌道)を返す
def ViterbiPath(PathWeightMap):
    Path = [[0,0] for t in range(T_horizon)]  #各tにおけるセル番号[x,y]


    return Path

#推定されたパスを（トピックかサービスで）送る
#def SendPath(Path):

#パスをファイル保存する（形式未定）
def SavePath(Path, Distance, outputname):
    print Path
    # 結果をファイル保存
    f = open( outputname + "_Path.csv" , "w")# , "sjis" )
    for i in range(len(Path)):
        f.write(Path[i] + ",")
        #f.write('\n')
    f.close()

#パス計算のために使用した確率値マップを（トピックかサービスで）送る
#def SendProbMap(PathWeightMap):

#パス計算のために使用した確率値マップをファイル保存する
def SaveProbMap(PathWeightMap, outputname):
    print PathWeightMap
    # 結果をファイル保存
    f = open( outputname + "_PathWeightMap.csv" , "w")# , "sjis" )
    for i in range(len(PathWeightMap)):
      for j in range(len(PathWeightMap[i])):
        f.write(PathWeightMap[i][j] + ",")
      f.write('\n')
    f.close()


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
            print W_list_sj,W_list[c] + " (one name)"
            
    print JuliusVer,HMMtype
    if (JuliusVer == "v4.4" and HMMtype == "DNN"):
      #hatsuonのすべての単語の音素表記を"*_I"にする
      for i in range(len(hatsuon)):
        hatsuon[i] = hatsuon[i].replace("_S","_I")
        hatsuon[i] = hatsuon[i].replace("_B","_I")
        hatsuon[i] = hatsuon[i].replace("_E","_I")
      
      #hatsuonの単語の先頭の音素を"*_B"にする
      for i in range(len(hatsuon)):
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

#パスの移動距離を計算、ファイル保存
def PathDistance(PathWeightMap):
    Distance = 0

    return Distance


########################################
if __name__ == '__main__':
    import sys
    #import os.path
    import time
    import random
    #import rospy
    #from std_msgs.msg import String
    from __init__ import *
    from JuliusNbest_dec import *

    #開始時刻を保持
    start_time = time.time()
    
    #学習済みパラメータフォルダ名を要求
    trialname = sys.argv[1]
    #print trialname
    #trialname = raw_input("trialname?(folder) >")

    #読み込むパーティクル番号を要求
    #particle_num = sys.argv[2] #0

    #重みファイルを読み込み
    for line in open(datafolder + trialname + '/'+ str(step) + '/weights.csv', 'r'):   ##読み込む
          MAX_Samp = int(line)
    #最大尤度のパーティクル番号を保存
    particle_num = MAX_Samp

    #ロボット初期位置の候補番号を要求
    init_position_num = sys.argv[3] #0

    #音声命令のファイル番号を要求   
    speech_num = sys.argv[4] #0

    ##FullPath of folder
    filename = datafolder + trialname + "/" + str(step) +"/"
    print filename, particle_num
    outputname = outputfolder + trialname + navigation_folder + "T"+T_horizon+"N"+N_best+"A"+Approx+"S"+init_position_num+"G"+speech_num

    Makedir( outputfolder + trialname )
    Makedir( outputfolder + trialname + navigation_folder )
    #Makedir( outputname )

    #学習済みパラメータの読み込み
    THETA = ReadParameters(particle_num, filename)
    #THETA = [W,W_index,Myu,Sig,pi,phi_l,K,L]
    W_index = THETA[1]
    
    ##単語辞書登録
    WordDictionaryUpdate2(step, filename, W_index)     

    ##コストマップの読み込み
    costmap = ReadCostMap()

    #音声認識開始時刻(初期化読み込み処理終了時刻)を保持
    start_recog_time = time.time()
    time_init = start_time - start_recog_time
    fp = open( outputname + "time_init.txt", 'a')
    fp.write(str(step)+","+str(time_init)+"\n")
    fp.close()

    ##音声ファイルを読み込み
    speech_file = ReadSpeech(speech_num)

    #音声認識
    S_Nbest = SpeechRecognition(speech_file, W_index, step, trialname, outputname)

    #音声認識終了時刻（PP開始時刻）を保持
    end_recog_time = time.time()
    time_recog = start_recog_time - end_recog_time
    fp = open( outputname + "time_recog.txt", 'a')
    fp.write(str(step)+","+str(time_recog)+"\n")
    fp.close()

    #パスプランニング
    Path, PathWeightMap = PathPlanner(S_Nbest, X_candidates[init_position_num], THETA, costmap)

    #PP終了時刻を保持
    end_pp_time = time.time()
    time_pp = end_pp_time - end_recog_time
    fp = open( outputname + "time_pp.txt", 'a')
    fp.write(str(step)+","+str(time_pp)+"\n")
    fp.close()

    #パスの移動距離
    Distance = PathDistance(Path)

    #パスを送る
    #SendPath(Path)
    #パスを保存
    SavePath(Path, Distance, outputname)


    #確率値マップを送る
    #SendProbMap(PathWeightMap)
    #確率値マップを保存
    SaveProbMap(PathWeightMap, outputname)


########################################

