#coding:utf-8

##############################################
##場所概念モデル
##SpCoNavi (on SIGVerse)のための学習用
##############################################


#---遂行タスクI(TAMD)---#

##↓別プログラムとして作成中
#相互推定のイテレーションごとに選ばれた学習結果の評価値（ARI、コーパスPAR、単語PAR）、（事後確率値）を出力
##単語PARはp(O_best|x_t)と正解を比較。x_tは正しいデータの平均値とする。

#↑のプログラムをインポートする

#---作業終了タスク（TAMD）---#
#sig_initパラメータを＿init＿.pyへ
#単語分割結果の読み込みの処理を修正
###pygameをインポート
#相互推定のイテレーションごとに位置分布の描画、保存
#データファイルを__init__.pyで指定するようにした
#パラメータ合わせる
#Turtlebotのデータに対応させる
#プログラムの整理（ちょっとだけ、無駄な計算や引数宣言の消去）
#sum()がnp.arrayの場合np.sumの方が高速(?)のため変更
#単語辞書が日本語音節表記でも音素ローマ字表記でも対応可能
#位置分布のμの値を保存するときに余計なものが混入していた（"_*"）のを修正済み

#---作業終了タスク---#
###正規化処理、0ワリ回避処理のコード短縮
###2次元ガウス分布の関数化、共分散をnumpy形式の行列計算に変更(時間があれば再確認)
#通常のMCLを平滑化MCLにする(結果確認済)
### Julius機能を消去。教示モードは動作をしながら教示時刻のみを保存する。(要確認)
#range()をxrange()に全て変更。xrange()の方が計算効率が良いらしい。
##ラグ値(LAG,lagu)、普通のMCL(ラグ値=0)の平滑化結果の出力を一つだけ(LAG)にする
##動作モデルと計測モデルの関数化->計測モデルの簡素化
###教示フェーズとMCLフェーズをわける
###センサ値、制御値を保存する
###保存したデータから平滑化MCLを動作させる
##全ての処理終了後に平滑化自己位置推定の結果をファイル出力(csv)
##認識発話単語集合をファイルへ出力
#パーティクル初期化の方法を変更(リスト内包表記いちいちしない)->いちいちリスト内表記！
##発話認識文(単語)データを読み込む
#<s>,<sp>,</s>を除く処理
#角度からラジアンへ変換する関数radian()はmathにあるためそちらに変更
##stick_breaking関数を見つけたので入れてみた
#多項分布の確率質量関数pmfを計算する関数を発見したので導入
##位置分布の描画用処理、viewerに送る位置分布サンプリング点群itiを設定
#robot初期状態を表す(x_init,y_init,radians(d_init))を作った
#motion_modelの関数名をsample_motion_modelへ変更。
#パーティクル数Mと位置分布平均Myuの引数名区別。Myuの初期化法修正
#0ワリ対処関数yudoupにyudo_sum==0の場合の例外処理を加えた。
###sampleではないmotion_modelを実装
#角度処理関数kakudoはPIの小数点精度問題があったため、より精度のよい修正版のkakudo2を作成。
#####XDt_true(真の角度値)とXDtが一致しない件について調査(角度が一周してることが判明)
######SBP,weak limit approximationについての確認、SBP他、初期化方法の修正
#myu0->m0,VV0->V0に修正
##動かさずにサンプル散らす関数sample_not_motion_modelを作った(挙動の確認する必要がある)
##最終学習結果の出力：初期値(*_init.csv)およびイテレーションごとにファイル出力
###データののっている要素のみをプリントする（データなしは表示しないようにしたが、ファイル出力するときは全部出す）
#パーティクルの平均を求める部分のコード短縮（ある程度できた）
#各サンプリングにおいて、データのない番号のものはどうする？：消す、表示しない、番号を前倒しにするか等
###各サンプリングにおいて、正しく正規化処理が行われているかチェック->たぶんOK
#####motion_modelの角度の例外処理->一応とりあえずやった
###ギブスサンプリングの収束判定条件は？(イテレートを何回するか)->とりあえず100回
###初期パラメータの値はどうするか？->とりあえずそこそこな感じにチューニングした
###どの要素をどの順番でサンプリングするか？->現状でとりあえずOK

#---保留---#
#計算速度の効率化は後で。
#いらないものは消す->少しは消した
##motionmodelで、パーティクルごとにおくるのではなく、群一気に行列としておくってnumpyで行列演算したほうが早いのでは？
##↑センサーモデルも同様？
#NormalInverseWishartDistribution関数のプログラムを見つけた。正確かどうか、どうなってるのか、要調査。
#余裕があれば場所概念ごとに色分けして位置分布を描画する(場所概念ごとの混合ガウス)
#Xtとμが遠いとg2の値がアンダーフローする可能性がある(logで計算すればよい？)問題があれば修正。
#ガウス分布を計算する関数をlogにする
#センサ値をintにせずにそのまま利用すればセンサ関係の尤度の値の計算精度があがるかも？
###動作モデルがガウスなので計算で求められるかもしれない件の数式導出


##ギブスサンプリング##
#W～ディリクレ＝マルチ*ディリクレ  L個：実装できたかな？
#μ、Σ～ガウス*ガウスウィシャート  K個：旧モデルの流用でok
#π～ディリクレ＝マルチ*GEM  1個：一応できた？GEM分布の事後分布の計算方法要確認
#Φ～ディリクレ＝マルチ*GEM  L個：同上
#Ct～多項値P(O|Wc)*多項値P(i|φc)*多項P(c|π)  N個：できた？
#it～ガウス値N(x|μk,Σk)*多項P(k|φc)  N個：できた？
#xt(no t)～計測モデル値*動作モデル値*動作モデルパーティクル (EndStep-N)個：概ねできた？
#xt(on t)～計測モデル値*動作モデル値*itの式(混合ガウス値)*動作モデルパーティクル N個：同上

import glob
import codecs
import re
import os
import sys
#import pygame
import random
import string
import numpy as np
from numpy.linalg import inv, cholesky
from scipy.stats import chi2
from math import pi as PI
from math import cos,sin,sqrt,exp,log,fabs,fsum,degrees,radians,atan2
from __init__ import *
from submodules import *

def gaussian(x,myu,sig):
    ###1次元ガウス分布
    gauss = (1.0 / sqrt(2.0*PI*sig*sig)) * exp(-1.0*(float((x-myu)*(x-myu))/(2.0*sig*sig)))
    return gauss
    
def gaussian2d(Xx,Xy,myux,myuy,sigma):
    ###ガウス分布(2次元)
    sqrt_inb = float(1) / ( 2.0 * PI * sqrt( np.linalg.det(sigma)) )
    xy_myu = np.array( [ [float(Xx - myux)],[float(Xy - myuy)] ] )
    dist = np.dot(np.transpose(xy_myu),np.linalg.solve(sigma,xy_myu))
    gauss2d = (sqrt_inb) * exp( float(-1/2) * dist )
    return gauss2d
    
def yudoup(yudo,yudo_sum): #float( 10 ** (-200) )
    if yudo_sum == 0 :  #エラー処理
        yudo = [0.1 for j in xrange(len(yudo))]
        yudo_sum = sum(yudo)
        print "yudo_sum is 0"
    if yudo_sum < 10**(-15) : #0.000000000000001: #+0000000000
        for j in xrange(len(yudo)):
          yudo[j] = yudo[j] * 10.0**12 #100000000000 #+00000
        yudo_sum = yudo_sum * 10.0**12 #100000000000 #+00000
        yudo,yudo_sum = yudoup(yudo,yudo_sum)
        print "yudoup!"
    return yudo,yudo_sum

def fill_param(param, default):   ##パラメータをNone の場合のみデフォルト値に差し替える関数
    if (param == None): return default
    else: return param

def invwishartrand_prec(nu,W):
    return inv(wishartrand(nu,W))

def invwishartrand(nu, W):
    return inv(wishartrand(nu, inv(W)))

def wishartrand(nu, W):
    dim = W.shape[0]
    chol = cholesky(W)
    #nu = nu+dim - 1
    #nu = nu + 1 - np.axrange(1,dim+1)
    foo = np.zeros((dim,dim))
    
    for i in xrange(dim):
        for j in xrange(i+1):
            if i == j:
                foo[i,j] = np.sqrt(chi2.rvs(nu-(i+1)+1))
            else:
                foo[i,j]  = np.random.normal(0,1)
    return np.dot(chol, np.dot(foo, np.dot(foo.T, chol.T)))

#http://nbviewer.ipython.org/github/fonnesbeck/Bios366/blob/master/notebooks/Section5_2-Dirichlet-Processes.ipynb
def stick_breaking(alpha, k):
    betas = np.random.beta(1, alpha, k)
    remaining_pieces = np.append(1, np.cumprod(1 - betas[:-1]))
    p = betas * remaining_pieces
    return p/p.sum()

#http://stackoverflow.com/questions/13903922/multinomial-pmf-in-python-scipy-numpy
class Multinomial(object):
  def __init__(self, params):
    self._params = params

  def pmf(self, counts):
    if not(len(counts)==len(self._params)):
      raise ValueError("Dimensionality of count vector is incorrect")

    prob = 1.
    for i,c in enumerate(counts):
      prob *= self._params[i]**counts[i]

    return prob * exp(self._log_multinomial_coeff(counts))

  def log_pmf(self,counts):
    if not(len(counts)==len(self._params)):
      raise ValueError("Dimensionality of count vector is incorrect")

    prob = 0.
    for i,c in enumerate(counts):
      prob += counts[i]*log(self._params[i])

    return prob + self._log_multinomial_coeff(counts)

  def _log_multinomial_coeff(self, counts):
    return self._log_factorial(sum(counts)) - sum(self._log_factorial(c)
                                                    for c in counts)

  def _log_factorial(self, num):
    if not round(num)==num and num > 0:
      raise ValueError("Can only compute the factorial of positive ints")
    return sum(log(n) for n in range(1,num+1))

def MI_binary(b,W,pi,c):  #Mutual information(二値版):word_index、W、π、Ct
    #相互情報量の計算
    POC = W[c][b] * pi[c] #Multinomial(W[c]).pmf(B) * pi[c]   #場所の名前の多項分布と場所概念の多項分布の積
    PO = sum([W[ct][b] * pi[ct] for ct in xrange(L)]) #Multinomial(W[ct]).pmf(B)
    PC = pi[c]
    POb = 1.0 - PO
    PCb = 1.0 - PC
    PObCb = PCb - PO + POC
    POCb = PO - POC
    PObC = PC - POC
    
    # 相互情報量の定義の各項を計算
    temp1 = POC * log(POC/(PO*PC), 2)
    temp2 = POCb * log(POCb/(PO*PCb), 2)
    temp3 = PObC * log(PObC/(POb*PC), 2)
    temp4 = PObCb * log(PObCb/(POb*PCb), 2)
    score = temp1 + temp2 + temp3 + temp4
    return score

def Mutual_Info(W,pi):  #Mutual information:W、π 
    MI = 0
    for c in xrange(len(pi)):
      PC = pi[c]
      for j in xrange(len(W[c])):
        #B = [int(i==j) for i in xrange(len(W[c]))]
        PO = fsum([W[ct][j] * pi[ct] for ct in xrange(len(pi))])  #Multinomial(W[ct]).pmf(B)
        POC = W[c][j] * pi[c]   #場所の名前の多項分布と場所概念の多項分布の積
        
        
        # 相互情報量の定義の各項を計算
        MI = MI + POC * ( log((POC/(PO*PC)), 2) )
    
    return MI


def position_data_read_pass(directory,DATA_NUM):
    all_position=[] 
    hosei = 1.5 # 04だけ*2, 06は-1, 10は*1.5

    for i in range(DATA_NUM):
            #if  (i in test_num)==False:
            f=directory+"/position/"+repr(i)+".txt"
            position=[] #(x,y,sin,cos)
            itigyoume = 1
            for line in open(f, 'r').readlines():
                if (itigyoume == 1):
                  data=line[:-1].split('	')
                  #print data
                  position +=[float(data[0])*(-1) + float(origin[0]*resolution)*hosei]
                  position +=[float(data[1])]
                  itigyoume = 0
            all_position.append(position)
    
    #座標系の返還
    #Xt = (np.array(all_position) + origin[0] ) / resolution #* 10
    return np.array(all_position)


"""
def Name_data_read(directory,word_increment,DATA_NUM):
    name_data_set=[]
    
    for i in range(DATA_NUM):
        name_data=[0 for w in range(len(name_list))]

        if  (i in test_num)==False:
            try:
                file=directory+Name_data_dir+repr(i)+".txt"
                data=np.genfromtxt(file, delimiter="\n", dtype='S' )
                #print file

                try:
                    for d in data:
                        #print d
                        for w,dictionry in enumerate(name_list):
                            if d == dictionry:
                                name_data[w]+=word_increment


                except TypeError:
                    #print d
                    for w,dictionry in enumerate(name_list):
                        if data == dictionry:
                            name_data[w]+=word_increment
            except IOError:
                pass
            name_data=np.array(name_data)
            name_data_set.append(name_data)
        else:
            print i
        #else:
            #print i,"is test data."
    return np.array(name_data_set)
"""

# Simulation
def simulate(iteration,filename):
    ##発話認識文(単語)データを読み込む
    ##空白またはカンマで区切られた単語を行ごとに読み込むことを想定する
    sample_num = 1  #取得するサンプル数
    #N = 0      #データ個数用
    #Otb = [[] for sample in xrange(sample_num)]   #音声言語情報：教示データ

    inputfile = inputfolder_SIG  + trialname
    filename  = outputfolder_SIG + trialname
    
    ##S## ##### Ishibushi's code #####
    env_para = np.genfromtxt(inputfile+"/Environment_parameter.txt",dtype= None,delimiter =" ")

    MAP_X = float(env_para[0][1])  #Max x value of the map
    MAP_Y = float(env_para[1][1])  #Max y value of the map
    map_x = float(env_para[2][1])  #Min x value of the map
    map_y = float(env_para[3][1])  #Max y value of the map

    map_center_x = ((MAP_X - map_x)/2)+map_x
    map_center_y = ((MAP_Y - map_x)/2)+map_y
    mu_0 = np.array([map_center_x,map_center_y,0,0])
    #mu_0_set.append(mu_0)
    DATA_initial_index = int(env_para[5][1]) #Initial data num
    DATA_last_index = int(env_para[6][1]) #Last data num
    DATA_NUM = DATA_last_index - DATA_initial_index +1
    ##E## ##### Ishibushi's code ######
    
    #DATA read
    pose = position_data_read_pass(inputfile,DATA_NUM)
    #name = Name_data_read(inputfile,word_increment,DATA_NUM)
    
    for sample in xrange(sample_num):
      #NN = 0
      N = 0
      Otb = []
      #テキストファイルを読み込み
      #for line in open(filename + '/out_gmm_' + str(iteration) + '/' + str(sample) + '_samp.100', 'r'):   ##*_samp.100を順番に読み込む
      for word_data_num in range(DATA_NUM):
        f = open(inputfile + "/name/per_100/word" + str(word_data_num) + ".txt", "r")
        line = f.read()
        #print line
        itemList = line[:-1].split(' ')
        
        #<s>,<sp>,</s>を除く処理：単語に区切られていた場合
        for b in xrange(5):
          if ("<s><s>" in itemList):
            itemList.pop(itemList.index("<s><s>"))
          if ("<s><sp>" in itemList):
            itemList.pop(itemList.index("<s><sp>"))
          if ("<s>" in itemList):
            itemList.pop(itemList.index("<s>"))
          if ("<sp>" in itemList):
            itemList.pop(itemList.index("<sp>"))
          if ("<sp><sp>" in itemList):
            itemList.pop(itemList.index("<sp><sp>"))
          if ("</s>" in itemList):
            itemList.pop(itemList.index("</s>"))
          if ("<sp></s>" in itemList):
            itemList.pop(itemList.index("<sp></s>"))
          if ("" in itemList):
            itemList.pop(itemList.index(""))
        #<s>,<sp>,</s>を除く処理：単語中に存在している場合
        for j in xrange(len(itemList)):
          itemList[j] = itemList[j].replace("<s><s>", "")
          itemList[j] = itemList[j].replace("<s>", "")
          itemList[j] = itemList[j].replace("<sp>", "")
          itemList[j] = itemList[j].replace("</s>", "")
        for b in xrange(5):
          if ("" in itemList):
            itemList.pop(itemList.index(""))
        
        #Otb[sample] = Otb[sample] + [itemList]
        Otb = Otb + [itemList]
        #if sample == 0:  #最初だけデータ数Nを数える
        N = N + 1  #count
        #else:
        #  Otb[] = Otb[NN] + itemList
        #  NN = NN + 1
        
        #for j in xrange(len(itemList)):
        #    print "%s " % (str(itemList[j])),
        #print ""  #改行用
      
      
      ##場所の名前の多項分布のインデックス用
      W_index = []
      for n in xrange(N):
        for j in xrange(len(Otb[n])):
          if ( (Otb[n][j] in W_index) == False ):
            W_index.append(Otb[n][j])
            #print str(W_index),len(W_index)
      
      print "[",
      for i in xrange(len(W_index)):
        print "\""+ str(i) + ":" + str(W_index[i]) + "\",",
      print "]"
      
      ##時刻tデータごとにBOW化(?)する、ベクトルとする
      Otb_B = [ [0 for i in xrange(len(W_index))] for n in xrange(N) ]
      
      
      for n in xrange(N):
        for j in xrange(len(Otb[n])):
          for i in xrange(len(W_index)):
            if (W_index[i] == Otb[n][j] ):
              Otb_B[n][i] = Otb_B[n][i] + word_increment
      #print Otb_B
      
      #N = DATA_NUM
      if N != DATA_NUM:
         print "DATA_NUM" + str(DATA_NUM) + ":KYOUJI error!! N:" + str(N)  ##教示フェーズの教示数と読み込んだ発話文データ数が違う場合
         #exit()
      
      #TN = [i for i in xrange(N)]#[0,1,2,3,4,5]  #テスト用
      
      ##教示位置をプロットするための処理
      #x_temp = []
      #y_temp = []
      #for t in xrange(len(TN)):
      #  x_temp = x_temp + [Xt[int(TN[t])][0]]  #設定は実際の教示時刻に対応できるようになっている。
      #  y_temp = y_temp + [Xt[int(TN[t])][1]]  #以前の設定のままで、動かせるようにしている。
      """
      EndStep = 0
      if (data_name != 'test000'):
        i = 0
        Xt = []
        #Xt = [(0.0,0.0) for n in xrange(len(HTW)) ]
        TN = []
        for line3 in open('./../sample/' + data_name, 'r'):
          itemList3 = line3[:-1].split(',')
          Xt = Xt + [(float(itemList3[0]), float(itemList3[1]))]
          TN = TN + [i]
          print TN
          i = i + 1
        
        #Xt = Xt_temp
        EndStep = len(Xt)-1
      """
      Xt = pose
      TN = [i for i in range(DATA_NUM)]
      
      
  ######################################################################
  ####                   ↓場所概念学習フェーズ↓                   ####
  ######################################################################
      #TN[N]：教示時刻(step)集合
      
      #Otb_B[N][W_index]：時刻tごとの発話文をBOWにしたものの集合
      
      ##各パラメータ初期化処理
      print u"Initialize Parameters..."
      #xtは既にある、ct,it,Myu,S,Wは事前分布からサンプリングにする？(要相談)
      Ct = [ int(n/15) for n in xrange(N)] #[0,0,1,1,2,3] random.uniform(0,L)    #物体概念のindex[N]
      It = [ int(n/15) for n in xrange(N)]#[1,1,2,2,3,2] random.uniform(0,K)    #位置分布のindex[N]
      ##領域範囲内に一様乱数
      #if (data_name == "test000"):
      Myu = [ np.array([[ int( random.uniform(WallXmin,WallXmax) ) ],[ int( random.uniform(WallYmin,WallYmax) ) ]]) for i in xrange(K) ]      #位置分布の平均(x,y)[K]
      #else:
      #  Myu = [ np.array([[ random.uniform(-37.8+5,-37.8+80-10) ],[ random.uniform(-34.6+5,-34.6+57.6-10) ]]) for i in xrange(K) ]      #位置分布の平均(x,y)[K]
      S = [ np.array([ [sig_init, 0.0],[0.0, sig_init] ]) for i in xrange(K) ]      #位置分布の共分散(2×2次元)[K]
      W = [ [beta0 for j in xrange(len(W_index))] for c in xrange(L) ]  #場所の名前(多項分布：W_index次元)[L]
      pi = stick_breaking(gamma, L)#[ 0 for c in xrange(L)]     #場所概念のindexの多項分布(L次元)
      phi_l = [ stick_breaking(alpha, K) for c in xrange(L) ]#[ [0 for i in xrange(K)] for c in xrange(L) ]  #位置分布のindexの多項分布(K次元)[L]
      
      
      print Myu
      print S
      print W
      print pi
      print phi_l
      
      ###初期値を保存(このやり方でないと値が変わってしまう)
      Ct_init = [Ct[n] for n in xrange(N)]
      It_init = [It[n] for n in xrange(N)]
      Myu_init = [Myu[i] for i in xrange(K)]
      S_init = [ np.array([ [S[i][0][0], S[i][0][1]],[S[i][1][0], S[i][1][1]] ]) for i in xrange(K) ]
      W_init = [W[c] for c in xrange(L)]
      pi_init = [pi[c] for c in xrange(L)]
      phi_l_init = [phi_l[c] for c in xrange(L)]
      
      
      
      
      ##場所概念の学習
      #関数にとばす->のは後にする
      print u"- <START> Learning of Location Concepts ver. NEW MODEL. -"
      
      for iter in xrange(num_iter):   #イテレーションを行う
        print 'Iter.'+repr(iter+1)+'\n'
        
        

        
        ########## ↓ ##### W(場所の名前：多項分布)のサンプリング ##### ↓ ##########
        ##ディリクレ多項からディリクレ事後分布を計算しサンプリングする
        ##ディリクレサンプリング関数へ入れ込む配列を作ればよい
        ##ディリクレ事前分布をサンプリングする必要はない->共役
        print u"Sampling Wc..."
        
        #data = [Otb_B[1],Otb_B[3],Otb_B[7],Otb_B[8]]  #仮データ
        
        #temp = np.ones((len(W_index),L))*beta0 #
        temp = [ [beta0 for j in xrange(len(W_index))] for c in xrange(L) ]  #集めて加算するための配列:パラメータで初期化しておけばよい
        #temp = [ np.ones(len(W_index))*beta0 for c in xrange(L)]
        #Ctがcであるときのデータを集める
        for c in xrange(L) :   #ctごとにL個分計算
          #temp = np.ones(len(W_index))*beta0
          nc = 0
          ##事後分布のためのパラメータ計算
          if c in Ct : 
            for t in xrange(N) : 
              if Ct[t] == c : 
                #データを集めるたびに値を加算
                for j in xrange(len(W_index)):    #ベクトル加算？頻度
                  temp[c][j] = temp[c][j] + Otb_B[t][j]
                nc = nc + 1  #データが何回加算されたか
              
          if (nc != 0):  #データなしのcは表示しない
            print "%d n:%d %s" % (c,nc,temp[c])
          
          #加算したデータとパラメータから事後分布を計算しサンプリング
          sumn = sum(np.random.dirichlet(temp[c],1000)) #fsumではダメ
          W[c] = sumn / sum(sumn)
          #print W[c]
        
        #Dir_0 = np.random.dirichlet(np.ones(L)*jp)
        #print Dir_0
        
        #ロバストなサンプリング結果を得るために
        #sumn = sum(np.random.dirichlet([0.1,0.2,0.5,0.1,0.1],10000))
        #multi = sumn / fsum(sumn)
        
        ########## ↑ ##### W(場所の名前：多項分布)のサンプリング ##### ↑ ##########
        
        ########## ↓ ##### μΣ(位置分布：ガウス分布の平均、共分散行列)のサンプリング ##### ↓ ##########
        print u"Sampling myu_i,Sigma_i..."
        #myuC = [ np.zeros((2,1)) for k in xrange(K) ] #np.array([[ 0.0 ],[ 0.0 ]])
        #sigmaC = [ np.zeros((2,2)) for k in xrange(K) ] #np.array([ [0,0],[0,0] ])
        np.random.seed()
        nk = [0 for j in xrange(K)]
        for j in xrange(K) : 
          ###jについて、Ctが同じものを集める
          #n = 0
          
          xt = []
          if j in It : 
            for t in xrange(N) : 
              if It[t] == j : 
                xt_To = TN[t]
                xt = xt + [ np.array([ [Xt[xt_To][0]], [Xt[xt_To][1]] ]) ]
                nk[j] = nk[j] + 1
          
          m_ML = np.array([[0.0],[0.0]])
          if nk[j] != 0 :        ##0ワリ回避
            m_ML = sum(xt) / float(nk[j]) #fsumではダメ
            print "n:%d m_ML.T:%s" % (nk[j],str(m_ML.T))
          
          #m0 = np.array([[0],[0]])   ##m0を元に戻す
          
          ##ハイパーパラメータ更新
          kappaN = kappa0 + nk[j]
          mN = ( (kappa0*m0) + (nk[j]*m_ML) ) / kappaN
          nuN = nu0 + nk[j]
          
          dist_sum = 0.0
          for k in xrange(nk[j]) : 
            dist_sum = dist_sum + np.dot((xt[k] - m_ML),(xt[k] - m_ML).T)
          VN = V0 + dist_sum + ( float(kappa0*nk[j])/(kappa0+nk[j]) ) * np.dot((m_ML - m0),(m_ML - m0).T)
          
          #if nk[j] == 0 :        ##0ワリ回避
          #  #nuN = nu0# + 1  ##nu0=nuN=1だと何故かエラーのため
          #  #kappaN = kappaN# + 1
          #  mN = np.array([[ int( random.uniform(1,WallX-1) ) ],[ int( random.uniform(1,WallY-1) ) ]])   ###領域内に一様
          
          ##3.1##Σを逆ウィシャートからサンプリング
          
          samp_sig_rand = np.array([ invwishartrand(nuN,VN) for i in xrange(100)])    ######
          samp_sig = np.mean(samp_sig_rand,0)
          #print samp_sig
          
          if np.linalg.det(samp_sig) < -0.0:
            samp_sig = np.mean(np.array([ invwishartrand(nuN,VN)]),0)
          
          ##3.2##μを多変量ガウスからサンプリング
          #print mN.T,mN[0][0],mN[1][0]
          x1,y1 = np.random.multivariate_normal([mN[0][0],mN[1][0]],samp_sig / kappaN,1).T
          #print x1,y1
          
          Myu[j] = np.array([[x1],[y1]])
          S[j] = samp_sig
          
        
        for j in xrange(K) : 
          if (nk[j] != 0):  #データなしは表示しない
            print 'myu'+str(j)+':'+str(Myu[j].T),
        print ''
        
        for j in xrange(K):
          if (nk[j] != 0):  #データなしは表示しない
            print 'sig'+str(j)+':'+str(S[j])
          
          
        """
        #データのあるKのみをプリントする？(未実装)
        print "myu1:%s myu2:%s myu3:%s myu4:%s myu5:%s" % (str(myuC[0].T), str(myuC[1].T), str(myuC[2].T),str(myuC[3].T), str(myuC[4].T))
        print "sig1:\n%s \nsig2:\n%s \nsig3:\n%s" % (str(sigmaC[0]), str(sigmaC[1]), str(sigmaC[2]))
        """
        #Myu = myuC
        #S = sigmaC
        
        ########## ↑ ##### μΣ(位置分布：ガウス分布の平均、共分散行列)のサンプリング ##### ↑ ##########
        
        
       ########## ↓ ##### π(場所概念のindexの多項分布)のサンプリング ##### ↓ ##########
        print u"Sampling PI..."
        
        #GEM = stick_breaking(gamma, L)
        #print GEM
        
        temp = np.ones(L) * (gamma / float(L)) #np.array([ gamma / float(L) for c in xrange(L) ])   #よくわからないので一応定義
        for c in xrange(L):
          temp[c] = temp[c] + Ct.count(c)
        #for t in xrange(N):    #Ct全データに対して
        #  for c in xrange(L):  #index cごとに
        #    if Ct[t] == c :      #データとindex番号が一致したとき
        #      temp[c] = temp[c] + 1
        #print temp  #確認済み
        
        #とりあえずGEMをパラメータとして加算してみる->桁落ちが発生していて意味があるのかわからない->パラメータ値を上げてみる&tempを正規化して足し合わせてみる(やめた)
        #print fsum(GEM),fsum(temp)
        #temp = temp / fsum(temp)
        #temp =  temp + GEM
        
        #持橋さんのスライドのやり方の方が正しい？ibis2008-npbayes-tutorial.pdf
        
        #print temp
        #加算したデータとパラメータから事後分布を計算しサンプリング
        sumn = sum(np.random.dirichlet(temp,1000)) #fsumではダメ
        pi = sumn / np.sum(sumn)
        print pi
        
        ########## ↑ ##### π(場所概念のindexの多項分布)のサンプリング ##### ↑ ##########
        
        
        ########## ↓ ##### φ(位置分布のindexの多項分布)のサンプリング ##### ↓ ##########
        print u"Sampling PHI_c..."
        
        #GEM = [ stick_breaking(alpha, K) for c in xrange(L) ]
        #print GEM
        
        for c in xrange(L):  #L個分
          temp = np.ones(K) * (alpha / float(K)) #np.array([ alpha / float(K) for k in xrange(K) ])   #よくわからないので一応定義
          #Ctとcが一致するデータを集める
          if c in Ct :
            for t in xrange(N):
              if Ct[t] == c:  #Ctとcが一致したデータで
                for k in xrange(K):  #index kごとに
                  if It[t] == k :      #データとindex番号が一致したとき
                    temp[k] = temp[k] + 1  #集めたデータを元に位置分布のindexごとに加算
            
          
          #ここからは一個分の事後GEM分布計算(πのとき)と同様
          #print fsum(GEM[c]),fsum(temp)
          #temp = temp / fsum(temp)
          #temp =  temp + GEM[c]
          
          #加算したデータとパラメータから事後分布を計算しサンプリング
          sumn = sum(np.random.dirichlet(temp,1000)) #fsumではダメ
          phi_l[c] = sumn / np.sum(sumn)
          
          if c in Ct:
            print c,phi_l[c]
          
          
        ########## ↑ ##### φ(位置分布のindexの多項分布)のサンプリング ##### ↑ ##########
        
        ########## ↓ ##### it(位置分布のindex)のサンプリング ##### ↓ ##########
        print u"Sampling it..."
        
        #It_B = [0 for k in xrange(K)] #[ [0 for k in xrange(K)] for n in xrange(N) ]   #多項分布のための出現回数ベクトル[t][k]
        #itと同じtのCtの値c番目のφc  の要素kごとに事後多項分布の値を計算
        temp = np.zeros(K)
        for t in xrange(N):    #時刻tごとのデータ
          phi_c = phi_l[int(Ct[t])]
          #np.array([ 0.0 for k in xrange(K) ])   #多項分布のパラメータ
          
          for k in xrange(K):
            #phi_temp = Multinomial(phi_c)
            #phi_temp.pmf([kのとき1のベクトル]) #パラメータと値は一致するのでphi_c[k]のままで良い
            
            #it=k番目のμΣについてのガウス分布をitと同じtのxtから計算
            xt_To = TN[t]
            g2 = gaussian2d(Xt[xt_To][0],Xt[xt_To][1],Myu[k][0],Myu[k][1],S[k])  #2次元ガウス分布を計算
            
            temp[k] = g2 * phi_c[k]
            #print g2,phi_c[k]  ###Xtとμが遠いとg2の値がアンダーフローする可能性がある
            
          temp = temp / np.sum(temp)  #正規化
          #print temp
          #Mult_samp = np.random.multinomial(1,temp)
          
          #print Mult_samp
          It_B = np.random.multinomial(1,temp) #Mult_samp [t]
          #print It_B[t]
          It[t] = np.where(It_B == 1)[0][0] #It_B.index(1)
          #for k in xrange(K):
          #  if (It_B[k] == 1):
          #    It[t] = k
          #    #print k
          
        #gaussian2d(Xx,Xy,myux,myuy,sigma)
        
        print It
        
        #多項分布からのサンプリング(1点)
        #http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.multinomial.html#numpy.random.multinomial
        #Mult_samp = np.random.multinomial(1,[確率の配列])
        ########## ↑ ##### it(位置分布のindex)のサンプリング ##### ↑ ##########
        
        
        ########## ↓ ##### Ct(場所概念のindex)のサンプリング ##### ↓ ##########
        print u"Sampling Ct..."
        #Ct～多項値P(Ot|Wc)*多項値P(it|φc)*多項P(c|π)  N個
        
        #It_B = [ [int(k == It[n]) for k in xrange(K)] for n in xrange(N) ]   #多項分布のための出現回数ベクトル[t][k]
        #Ct_B = [0 for c in xrange(L)] #[ [0 for c in xrange(L)] for n in xrange(N) ]   #多項分布のための出現回数ベクトル[t][l]
        
        temp = np.zeros(L)
        for t in xrange(N):    #時刻tごとのデータ
          #for k in xrange(K):
          #  if (k == It[t]):
          #    It_B[t][k] = 1
          
          #print It_B[t] #ok
          
          #np.array([ 0.0 for c in xrange(L) ])   #多項分布のパラメータ
          for c in xrange(L):  #場所概念のindexの多項分布それぞれについて
            #phi_temp = Multinomial(phi_l[c])
            W_temp = Multinomial(W[c])
            #print pi[c], phi_temp.pmf(It_B[t]), W_temp.pmf(Otb_B[t])
            temp[c] = pi[c] * phi_l[c][It[t]] * W_temp.pmf(Otb_B[t])    # phi_temp.pmf(It_B[t])各要素について計算
          
          temp = temp / np.sum(temp)  #正規化
          #print temp
          #Mult_samp = np.random.multinomial(1,temp)
          
          #print Mult_samp
          Ct_B = np.random.multinomial(1,temp) #Mult_samp
          #print Ct_B[t]
          
          Ct[t] = np.where(Ct_B == 1)[0][0] #Ct_B.index(1)
          #for c in xrange(L):
          #  if (Ct_B[c] == 1):
          #    Ct[t] = c
          #    #print c
          
        print Ct
        ########## ↑ ##### Ct(場所概念のindex)のサンプリング ##### ↑ ##########
        
        """
        ########## ↓ ##### xt(教示時刻で場合分け)のサンプリング ##### ↓ ##########
        print u"Sampling xt..."
        robot.input ((0,0))
        robot.move  (0, 0)
        
        
        #It_B = [ [0 for k in xrange(K)] for n in xrange(N) ]   #多項分布のための出現回数ベクトル[t][k]
        #
        #for t in xrange(N):    #時刻tごとのデータ
        #  for k in xrange(K):
        #    if (k == It[t]):
        #      It_B[t][k] = 1
        
        #It_1 = [ [(i==j)*1 for i in xrange(L)] for j in xrange(L)]   #i==jの要素が1．それ以外は0のベクトル
        
        #for t in xrange(EndStep):
        t = -1#EndStep-1
        while (t >= 0):
          ##t in Toかどうか関係ない部分の処理
          Xx_temp,Xy_temp,Xd_temp = [],[],[]
          yudo = []
          
          input1,input2 = Ut[t][0],Ut[t][1]
          robot.input ((input1,input2))
          robot.move  (d_trans, d_rot)
          d_trans = input2 * robot.bias_turn      #
          d_rot = radians(robot.bias_go) * input1  #
          #print t
          if (t+1 < EndStep):
            d_trans2 = Ut[t+1][1] * robot.bias_turn      #
            d_rot2 = radians(robot.bias_go) * Ut[t+1][0]  #
          
          for i in xrange(M):   ##全てのパーティクルに対し
            #動作モデルによりt-1からtの予測分布をサンプリング
            #動作モデル(表5.6)##↓###################################################ok
            if (t == 0):
              #xd,yd,sitad = sample_motion_model(d_rot,d_trans,para,Xinit[0],Xinit[1],Xinit[2]) #初期値を与えてよいのか？
              #xd,yd,sitad = Xt[t][0],Xt[t][1],XDt[t]  #最初の推定結果をそのまま用いる場合
              xd,yd,sitad = sample_not_motion_model(d_rot,d_trans,para_s,Xt[t][0],Xt[t][1],XDt[t]) #動かさずに粒子を散らすだけ
            else:
              xd,yd,sitad = sample_motion_model(d_rot,d_trans,para_s,Xt[t-1][0],Xt[t-1][1],XDt[t-1])
            Xx_temp = Xx_temp + [xd]
            Xy_temp = Xy_temp + [yd]
            Xd_temp = Xd_temp + [sitad]
            #動作モデル##↑###################################################
            
            #計測モデルを計算
            #尤度(重み)計算##↓###########################################
            #ロボットの姿勢、センサー値(地図)、パーティクルのセンサー値(計測)
            #各パーティクルにロボット飛ばす->センサー値を取得 をパーティクルごとに繰り返す
            yudo = yudo + [sensor_model(robot,xd,yd,sitad,sig_hit2,Zt[t])]
            #尤度(重み)計算##↑###########################################
            
            
          ###一回正規化してから尤度かけるようにしてみる
          #正規化処理
          #yudo_sum = fsum(yudo)
          #yudo,yudo_sum = yudoup(yudo,yudo_sum)     ####とても小さな浮動小数値をある程度まで大きくなるまで桁をあげる
          ###0ワリ対処処理
          #yudo_max = max(yudo)  #最大尤度のパーティクルを探す
          #yudo_summax = float(yudo_sum) / yudo_max
          #for j in xrange(M):
          #  yudo[j] = float(float(yudo[j])/yudo_max) / yudo_summax
          #  
          #  
          #for i in xrange(M):   ##全てのパーティクルに対し
            
            
            #動作モデル(t+1)尤度計算
            if (t+1 < EndStep):
              #print yudo[i],motion_model(d_rot2,d_trans2,para,Xt[t+1][0],Xt[t+1][1],XDt[t+1],xd,yd,sitad)
              yudo[i] = yudo[i] * motion_model(d_rot2,d_trans2,para_s,Xt[t+1][0],Xt[t+1][1],XDt[t+1],xd,yd,sitad)
            
            #tによって場合分け処理
            for n in xrange(N):
              if TN[n] == t:  #t in To
                #ガウス×多項 / Σ(ガウス×多項)-> ガウス / Σ(ガウス×多項)
                GM_sum = 0.0
                #print t
                #分母：混合ガウス部分の計算
                #phi_temp = Multinomial(phi_l[Ct[n]])
                for j in xrange(K):  #it=jごとのすべての位置分布において
                  ##パーティクルごとに計算する必要がある、パーティクルごとに値をもっていないといけない？
                  
                  g2 = gaussian2d(xd,yd,Myu[j][0],Myu[j][1],S[j])  #2次元ガウス分布を計算
                  GM_sum = GM_sum + g2 * phi_l[Ct[n]][j]    #各要素について計算
                  #phi_temp.pmf( It_1[j] )
                  
                  ##
                if (GM_sum != 0):
                  yudo[i] = yudo[i] * gaussian2d(xd,yd,Myu[It[n]][0],Myu[It[n]][1],S[It[n]]) / GM_sum
                #print yudo[i]
            
            
          ##推定状態確認用
          #MAINCLOCK.tick(FPS)
          events = pygame.event.get()
          for event in events:
                if event.type == KEYDOWN:
                    if event.key  == K_ESCAPE: exit()
          robot.set_position(Xt_true[t][0],Xt_true[t][1],XDt_true[t])
          robot.input ((0,0))
          robot.move  (0, 0)
          viewer.show(world,[[0,0]],M,Xx_temp,Xy_temp)
          
          
          #正規化処理
          yudo_sum = fsum(yudo)
          yudo,yudo_sum = yudoup(yudo,yudo_sum)     ####とても小さな浮動小数値をある程度まで大きくなるまで桁をあげる
          ###0ワリ対処処理
          yudo_max = max(yudo)  #最大尤度のパーティクルを探す
          yudo_summax = float(yudo_sum) / yudo_max
          for j in xrange(M):
            yudo[j] = float(float(yudo[j])/yudo_max) / yudo_summax
          
          #リサンプリング処理(一点のみ)
          ###確率サイコロ
          rand_c = random.random()        # Random float x, 0.0 <= x < 1.0
          #print rand_c
          pc_num = 0.0
          for i in xrange(M) : 
            pc_num = pc_num + yudo[i]
            if pc_num >= rand_c : 
              print t,int(Xt[t][0]),int(Xt[t][1]),int(degrees(XDt[t]))  #変更反映前のXtの確認用
              Xt[t] = (Xx_temp[i],Xy_temp[i])  #タプルの要素ごとに代入はできないため、タプルとして一気に代入
              XDt[t] = Xd_temp[i]
              rand_c = 1.1
          
          print t,int(Xt[t][0]),int(Xt[t][1]),int(degrees(XDt[t]))
          print t,int(Xt_true[t][0]),int(Xt_true[t][1]),degrees(XDt_true[t]),degrees(XDt_true[t])-360
          
          t = t-1
          
          #if t == -1:  ##動作確認用の無限ループ
          #    t = EndStep-1
        ########## ↑ ##### xt(教示時刻で場合分け)のサンプリング ##### ↑ ##########
        """
        
        """
        loop = 0
        if loop == 1:
          #サンプリングごとに各パラメータ値を出力
          fp = open('./data/' + filename + '/' + filename +'_samp'+ repr(iter)+'.csv', 'w')
          fp.write('sampling_data,'+repr(iter)+'\n')  #num_iter = 10  #イテレーション回数
          fp.write('Ct\n')
          for i in xrange(N):
            fp.write(repr(i)+',')
          fp.write('\n')
          for i in xrange(N):
            fp.write(repr(Ct[i])+',')
          fp.write('\n')
          fp.write('It\n')
          for i in xrange(N):
            fp.write(repr(i)+',')
          fp.write('\n')
          for i in xrange(N):
            fp.write(repr(It[i])+',')
          fp.write('\n')
          fp.write('Position distribution\n')
          for k in xrange(K):
            fp.write('Myu'+repr(k)+','+repr(Myu[k][0])+','+repr(Myu[k][1])+'\n')
          for k in xrange(K):
            fp.write('Sig'+repr(k)+'\n')
            fp.write(repr(S[k])+'\n')
          for c in xrange(L):
            fp.write('W'+repr(c)+','+repr(W[c])+'\n')
          for c in xrange(L):
            fp.write('phi_l'+repr(c)+','+repr(phi_l[c])+'\n')
          fp.write('pi'+','+repr(pi)+'\n')
          fp.close()
          fp_x = open( filename + '/' + filename +'_xt'+ repr(iter)+'.csv', 'w')
          for t in xrange(EndStep) : 
            fp_x.write(repr(Xt[t][0]) + ', ' + repr(Xt[t][1]) + '\n')
          fp_x.close()
        """
      
      
  ######################################################################
  ####                   ↑場所概念学習フェーズ↑                   ####
  ######################################################################
      
      
      loop = 1
      ########  ↓ファイル出力フェーズ↓  ########
      if loop == 1:
        print "--------------------"
        #最終学習結果を出力
        print u"\n- <COMPLETED> Learning of Location Concepts ver. NEW MODEL. -"
        print 'Sample: ' + str(sample)
        print 'Ct: ' + str(Ct)
        print 'It: ' + str(It)
        for c in xrange(L):
          print "W%d: %s" % (c,W[c])
        for k in xrange(K):
          print "myu%d: %s" % (k, str(Myu[k].T))
        for k in xrange(K):
          print "sig%d: \n%s" % (k, str(S[k]))
        print 'pi: ' + str(pi)
        for c in xrange(L):
          print 'phi' + str(c) + ':',
          print str(phi_l[c])
        
        print "--------------------"
        
        #サンプリングごとに各パラメータ値を出力
        if loop == 1:
          fp = open( filename + '/' + trialname +'_kekka_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
          fp.write('sampling_data,'+repr(iter+1)+'\n')  #num_iter = 10  #イテレーション回数
          fp.write('Ct\n')
          for i in xrange(N):
            fp.write(repr(i)+',')
          fp.write('\n')
          for i in xrange(N):
            fp.write(repr(Ct[i])+',')
          fp.write('\n')
          fp.write('It\n')
          for i in xrange(N):
            fp.write(repr(i)+',')
          fp.write('\n')
          for i in xrange(N):
            fp.write(repr(It[i])+',')
          fp.write('\n')
          fp.write('Position distribution\n')
          for k in xrange(K):
            fp.write('Myu'+repr(k)+','+repr(Myu[k][0][0])+','+repr(Myu[k][1][0])+'\n')
          for k in xrange(K):
            fp.write('Sig'+repr(k)+'\n')
            fp.write(repr(S[k])+'\n')
          
          for c in xrange(L):
            fp.write(',')
            for i in xrange(len(W_index)):
              fp.write(W_index[i] + ',')   #####空白が入っているものがあるので注意(', ')
            fp.write('\n')
            fp.write('W'+repr(c)+',')
            for i in xrange(len(W_index)):
              fp.write(repr(W[c][i])+',')
            fp.write('\n')
          for c in xrange(L):
            fp.write(',')
            for k in xrange(K):
              fp.write(repr(k)+',')
            fp.write('\n')
            fp.write('phi_l'+repr(c)+',')
            for k in xrange(K):
              fp.write(repr(phi_l[c][k])+',')
            fp.write('\n')
          fp.write(',')
          for c in xrange(L):
            fp.write(repr(c)+',')
          fp.write('\n')
          fp.write('pi'+',')
          for c in xrange(L):
            fp.write(repr(pi[c])+',')
          fp.write('\n')
          fp.close()
          #fp_x = open( filename + '/' + filename +'_xt'+ repr(iter)+'.csv', 'w')
          #for t in xrange(EndStep) : 
          #  fp_x.write(repr(Xt[t][0]) + ', ' + repr(Xt[t][1]) + '\n')
          #fp_x.close()
        
        
        
        
        #各パラメータ値、初期値を出力
        fp_init = open( filename + '/' + trialname + '_init_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        fp_init.write('init_data\n')  #num_iter = 10  #イテレーション回数
        fp_init.write('L,'+repr(L)+'\n')
        fp_init.write('K,'+repr(K)+'\n')
        fp_init.write('alpha,'+repr(alpha)+'\n')
        fp_init.write('gamma,'+repr(gamma)+'\n')
        fp_init.write('bata0,'+repr(beta0)+'\n')
        fp_init.write('kappa0,'+repr(kappa0)+'\n')
        fp_init.write('m0,'+repr(m0)+'\n')
        fp_init.write('V0,'+repr(V0)+'\n')
        fp_init.write('nu0,'+repr(nu0)+'\n')
        fp_init.write('sigma_init,'+repr(sig_init)+'\n')
        #fp_init.write('M,'+repr(M)+'\n')
        fp_init.write('N,'+repr(N)+'\n')
        fp_init.write('TN,'+repr(TN)+'\n')
        fp_init.write('Ct_init\n')
        for i in xrange(N):
          fp_init.write(repr(i)+',')
        fp_init.write('\n')
        for i in xrange(N):
          fp_init.write(repr(Ct_init[i])+',')
        fp_init.write('\n')
        fp_init.write('It_init\n')
        for i in xrange(N):
          fp_init.write(repr(i)+',')
        fp_init.write('\n')
        for i in xrange(N):
          fp_init.write(repr(It_init[i])+',')
        fp_init.write('\n')
        fp_init.write('Position distribution_init\n')
        for k in xrange(K):
          fp_init.write('Myu_init'+repr(k)+','+repr(Myu_init[k][0])+','+repr(Myu_init[k][1])+'\n')
        for k in xrange(K):
          fp_init.write('Sig_init'+repr(k)+'\n')
          fp_init.write(repr(S_init[k])+'\n')
        for c in xrange(L):
          fp_init.write('W_init'+repr(c)+','+repr(W_init[c])+'\n')
        #for c in xrange(L):
        #  fp_init.write('phi_l_init'+repr(c)+','+repr(phi_l_init[c])+'\n')
        #fp_init.write('pi_init'+','+repr(pi_init)+'\n')
        for c in xrange(L):
          fp_init.write(',')
          for k in xrange(K):
            fp_init.write(repr(k)+',')
          fp_init.write('\n')
          fp_init.write('phi_l_init'+repr(c)+',')
          for k in xrange(K):
            fp_init.write(repr(phi_l_init[c][k])+',')
          fp_init.write('\n')
        fp_init.write(',')
        for c in xrange(L):
          fp_init.write(repr(c)+',')
        fp_init.write('\n')
        fp_init.write('pi_init'+',')
        for c in xrange(L):
          fp_init.write(repr(pi_init[c])+',')
        fp_init.write('\n')
        
        fp_init.close()
        
        ##自己位置推定結果をファイルへ出力
        #filename_xt = raw_input("Xt:filename?(.csv) >")  #ファイル名を個別に指定する場合
        #filename_xt = filename
        #fp = open( filename + '/' + filename_xt + '_xt_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        #fp2 = open('./data/' + filename_xt + '_xt_true.csv', 'w')
        #fp3 = open('./data/' + filename_xt + '_xt_heikatsu.csv', 'w')
        #fp.write(Xt)
        #for t in xrange(EndStep) : 
        #    fp.write(repr(Xt[t][0]) + ', ' + repr(Xt[t][1]) + '\n')
        #    #fp2.write(repr(Xt_true[t][0]) + ', ' + repr(Xt_true[t][1]) + '\n')
        #    #fp2.write(repr(Xt_heikatsu[t][0]) + ', ' + repr(Xt_heikatsu[t][1]) + '\n')
        #fp.writelines(repr(Xt))
        #fp.close()
        #fp2.close()
        #fp3.close()
        
        ##認識発話単語集合をファイルへ出力
        #filename_ot = raw_input("Otb:filename?(.csv) >")  #ファイル名を個別に指定する場合
        filename_ot = trialname
        fp = open(filename + '/' + filename_ot + '_ot_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        fp2 = open(filename + '/' + filename_ot + '_w_index_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for n in xrange(N) : 
            for j in xrange(len(Otb[n])):
                fp.write(Otb[n][j] + ',')
            fp.write('\n')
        for i in xrange(len(W_index)):
            fp2.write(repr(i) + ',')
        fp2.write('\n')
        for i in xrange(len(W_index)):
            fp2.write(W_index[i] + ',')
        fp.close()
        fp2.close()
        
        print 'File Output Successful!(filename:'+filename+ "_" +str(iteration) + "_" + str(sample) + ')\n'
      
      
      ##パラメータそれぞれをそれぞれのファイルとしてはく
      if loop == 1:
        fp = open( filename + '/' + trialname + '_Myu_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for k in xrange(K):
          fp.write(repr(float(Myu[k][0][0]))+','+repr(float(Myu[k][1][0])) + '\n')
        fp.close()
        fp = open( filename + '/' + trialname + '_S_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for k in xrange(K):
          fp.write(repr(S[k][0][0])+','+repr(S[k][0][1])+','+repr(S[k][1][0]) + ','+repr(S[k][1][1])+'\n')
        fp.close()
        fp = open( filename + '/' + trialname + '_W_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for c in xrange(L):
          for i in xrange(len(W_index)):
            fp.write(repr(W[c][i])+',')
          fp.write('\n')
          #fp.write(repr(W[l][0])+','+repr(W[l][1])+'\n')
        fp.close()
        fp = open( filename + '/' + trialname + '_phi_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for c in xrange(L):
          for k in xrange(K):
            fp.write(repr(phi_l[c][k])+',')
          fp.write('\n')
        fp.close()
        fp = open( filename + '/' + trialname + '_pi_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for c in xrange(L):
          fp.write(repr(pi[c])+',')
        fp.write('\n')
        fp.close()
        
        fp = open( filename + '/' + trialname + '_Ct_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for t in xrange(N):
          fp.write(repr(Ct[t])+',')
        fp.write('\n')
        fp.close()
        
        fp = open( filename + '/' + trialname + '_It_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for t in xrange(N):
          fp.write(repr(It[t])+',')
        fp.write('\n')
        fp.close()

        #fp = open( filename + "/W_list.csv", 'w')
        #for w in xrange(len(W_index)):
        #  fp.write(W_index[w]+",")
        #fp.close()

      
      ########  ↑ファイル出力フェーズ↑  ########
      
      """
      ##学習後の描画用処理
      iti = []    #位置分布からサンプリングした点(x,y)を保存する
      #Plot = 500  #プロット数
      
      K_yes = 0
      ###全てのパーティクルに対し
      for j in range(K) : 
        yes = 0
        for t in xrange(N):  #jが推定された位置分布のindexにあるか判定
          if j == It[t]:
            yes = 0 #1
        if yes == 1:
          K_yes = K_yes + 1
          for i in xrange(Plot):
            if (data_name != "test000"):
              S_temp = [[ S[j][0][0]/(0.05*0.05) , S[j][0][1]/(0.05*0.05) ] , [ S[j][1][0]/(0.05*0.05) , S[j][1][1]/(0.05*0.05) ]]
              x1,y1 = np.random.multivariate_normal( [(Myu[j][0][0][0]+37.8)/0.05, (Myu[j][1][0][0]+34.6)/0.05] , S_temp , 1).T
            else:
              x1,y1 = np.random.multivariate_normal([Myu[j][0][0][0],Myu[j][1][0][0]],S[j],1).T
            #print x1,y1
            iti = iti + [[x1,y1]]
      
      #iti = iti + [[K_yes,Plot]]  #最後の要素[[位置分布の数],[位置分布ごとのプロット数]]
      #print iti
      filename2 = str(iteration) + "_" + str(sample)
      """
      
if __name__ == '__main__':
    import sys
    import os.path
    from __init__ import *
    #from JuliusLattice_dec import *
    #import time
    
    
    trialname = sys.argv[1]
    print trialname
    
    #出力ファイル名を要求
    #filename = raw_input("trialname?(folder) >")
    #start_time = time.time()
    #iteration_time = [0.0 for i in range(ITERATION)]
    filename = outputfolder_SIG + trialname
    Makedir( filename )
    #Makedir( "data/" + filename + "/lattice" )
    
    #p0 = os.popen( "PATH=$PATH:../../latticelm" )  #パスを通す-＞通らなかった
    
    for i in xrange(ITERATION):
      print "--------------------------------------------------"
      print "ITERATION:",i+1
      #start_iter_time = time.time()
      
      #Julius_lattice(i,filename)    ##音声認識、ラティス形式出力、opemFST形式へ変換
      #p = os.popen( "python JuliusLattice_gmm.py " + str(i+1) +  " " + filename )
      
      
      
      #while (os.path.exists("./data/" + filename + "/fst_gmm_" + str(i+1) + "/" + str(kyouji_count-1).zfill(3) +".fst" ) != True):
      #  print "./data/" + filename + "/fst_gmm_" + str(i+1) + "/" + str(kyouji_count-1).zfill(3) + ".fst",os.path.exists("./data/" + filename + "/fst_gmm_" + str(i+1).zfill(3) + "/" + str(kyouji_count-1) +".fst" ),"wait(60s)... or ERROR?"
      #  time.sleep(60.0) #sleep(秒指定)
      #print "ITERATION:",i+1," Julius complete!"
      """
      #for sample in xrange(sample_num):
      sample = 0  ##latticelmのパラメータ通りだけサンプルする
      for p1 in xrange(len(knownn)):
        for p2 in xrange(len(unkn)):
          if sample < sample_num:
            print "latticelm run. sample_num:" + str(sample)
            p = os.popen( "latticelm -input fst -filelist data/" + filename + "/fst_gmm_" + str(i+1) + "/fstlist.txt -prefix data/" + filename + "/out_gmm_" + str(i+1) + "/" + str(sample) + "_ -symbolfile data/" + filename + "/fst_gmm_" + str(i+1) + "/isyms.txt -burnin 100 -samps 100 -samprate 100 -knownn " + str(knownn[p1]) + " -unkn " + str(unkn[p2]) )   ##latticelm  ## -annealsteps 10 -anneallength 15
            time.sleep(1.0) #sleep(秒指定)
            while (os.path.exists("./data/" + filename + "/out_gmm_" + str(i+1) + "/" + str(sample) + "_samp.100" ) != True):
              print "./data/" + filename + "/out_gmm_" + str(i+1) + "/" + str(sample) + "_samp.100",os.path.exists("./data/" + filename + "/out_gmm_" + str(i+1) + "/" + str(sample) + "_samp.100" ),"wait(30s)... or ERROR?"
              p.close()
              p = os.popen( "latticelm -input fst -filelist data/" + filename + "/fst_gmm_" + str(i+1) + "/fstlist.txt -prefix data/" + filename + "/out_gmm_" + str(i+1) + "/" + str(sample) + "_ -symbolfile data/" + filename + "/fst_gmm_" + str(i+1) + "/isyms.txt -burnin 100 -samps 100 -samprate 100 -knownn " + str(knownn[p1]) + " -unkn " + str(unkn[p2]) )   ##latticelm  ## -annealsteps 10 -anneallength 15
              
              time.sleep(3.0) #sleep(秒指定)
            sample = sample + 1
            p.close()
      print "ITERATION:",i+1," latticelm complete!"
      """
      simulate(i+1,trialname)          ##場所概念の学習
      
      print "ITERATION:",i+1," Learning complete!"
      #sougo(i+1)             ##相互情報量計算+##単語辞書登録
      #print "ITERATION:",i+1," Language Model update!"
      #Language_model_update(i+1)  ##単語辞書登録
      #end_iter_time = time.time()
      #iteration_time[i] = end_iter_time - start_iter_time
    
    ##ループ後処理
    
    #p0.close()
    #end_time = time.time()
    #time_cost = end_time - start_time
    """
    fp = open('./data/' + filename + '/time.txt', 'w')
    fp.write(str(time_cost)+"\n")
    fp.write(str(start_time)+","+str(end_time)+"\n")
    for i in range(ITERATION):
      fp.write(str(i+1)+","+str(iteration_time[i])+"\n")
    """

########################################

