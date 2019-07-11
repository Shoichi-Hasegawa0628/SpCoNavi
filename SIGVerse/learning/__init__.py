#coding:utf-8
#パラメータ設定ファイル
import numpy as np
#SpCoNavi (on SIGVerse)のための学習用（learn4_3.py対応）

##### NEW #####
inputfolder_SIG  = "/mnt/hgfs/Dropbox/SpCoNavi/CoRL/dataset/similar/3LDK/"  #"/home/akira/Dropbox/SpCoNavi/data/"
outputfolder_SIG = "/mnt/hgfs/Dropbox/SpCoNavi/CoRL/data/"  #"/home/akira/Dropbox/SpCoNavi/data/"
#Navigation folder (他の出力ファイルも同フォルダ)
navigation_folder = "/navi/"  #outputfolder + trialname + / + navigation_folder + contmap.csv

#地図のyamlファイルと同じ値にする
resolution = 0.1 #0.050000
origin =  np.array([-10.000000, -10.000000]) #, 0.000000] #np.array([-30.000000, -20.000000]) #, 0.000000]


word_increment = 10

####################パラメータ####################
kyouji_count = 50 #100 #教示数をカウントする
M = 2000   #パーティクルの数(学習の条件と同じ：300、旧モデルと同じ：300)
#LAG = 100 + 1  ##(平滑化のラグ数 + 1)個の要素を持つラグ配列の要素数

#外壁座標
#WallX = 1600
#WallY = 1152#
WallXmin = -10
WallXmax = 10
WallYmin = 10
WallYmax = -10

###動作モデルパラメータ###(表5.6)
#para1 = 0.01  #0.50
#para2 = 0.01  #0.05
#para3 = 0.2   #0.8
#para4 = 0.5  #20.0
#para_s = [0,para1,para2,para3,para4] #最初の0は配列番号とpara番号を合わせるためのもの

###計測モデルパラメータ###
#sig_hit2 = 2.0  #パラメータに注意。元の設定：3


##初期(ハイパー)パラメータ
num_iter = 10          #場所概念学習のイテレーション回数
L = 10 #100                  #場所概念の数50#
K = 10 #100                  #位置分布の数50#
alpha = 1.0 #0.1 #10.0 #5.0#1.5#10.0               #位置分布のindexの多項分布のハイパーパラメータ1.5#
gamma = 1.0 #20.0 #15.0#8.0#20.0               #場所概念のindexの多項分布のハイパーパラメータ8.0#
beta0 = 0.1 #0.4#0.2               #場所の名前Wのハイパーパラメータ0.5#
kappa0 = 1e-3                #μのパラメータ、旧モデルのbeta0であることに注意
m0 = np.array([[0.0],[0.0]])   #μのパラメータ
V0 = np.eye(2)*2 #*1000              #Σのパラメータ
nu0 = 3.0 #3.0                    #Σのパラメータ、旧モデルでは1としていた(自由度の問題で2の方が良い?)、事後分布の計算のときに1加算していた

sig_init =  1.0 

##latticelmパラメータ
knownn = [2,3,4] #[3]#         #言語モデルのn-gram長 (3)
unkn = [3,4] #[3]#            #綴りモデルのn-gram長 (3),5
annealsteps = [3,5,10]    #焼き鈍し法のステップ数 (3)
anneallength = [5,10,15]  #各焼き鈍しステップのイタレーション数 (5)


##相互推定に関するパラメータ
sample_num = 1 #len(knownn)*len(unkn)  #取得するサンプル数
ITERATION = 1  #相互推定のイテレーション回数

##単語の選択の閾値
threshold = 0.01


#Plot = 2000#1000  #位置分布ごとの描画の点プロット数

#N_best_number = 10 #n-bestのnをどこまでとるか（n<=10）


#Juliusパラメータ
#Juliusフォルダのsyllable.jconf参照
JuliusVer = "v4.4" #"v.4.3.1" #
HMMtype = "DNN"  #"GMM"
lattice_weight = "AMavg"  #"exp" #音響尤度(対数尤度："AMavg"、尤度："exp")
wight_scale = -1.0

if (JuliusVer ==  "v4.4"):
  Juliusfolder = "/home/akira/Dropbox/Julius/dictation-kit-v4.4/"
else:
  Juliusfolder = "/home/akira/Dropbox/Julius/dictation-kit-v4.3.1-linux/"

if (HMMtype == "DNN"):
  lang_init = 'syllableDNN.htkdic' 
else:
  lang_init = 'web.000.htkdic' # 'trueword_syllable.htkdic' #'phonemes.htkdic' # 初期の単語辞書（./lang_mフォルダ内）
lang_init_DNN = 'syllableDNN.htkdic' #なごり

####################ファイル####################
speech_folder = "/home/*/Dropbox/Julius/directory/SpCoSLAM/*.wav" #*.wav" #音声の教示データフォルダ(Ubuntuフルパス)
speech_folder_go = "/home/akira/Dropbox/Julius/directory/SpCoSLAMgo/*.wav" #*.wav" #音声の教示データフォルダ(Ubuntuフルパス)
data_name = 'SpCoSLAM.csv'      # 'test000' #位置推定の教示データ(./../sampleフォルダ内)
lmfolder = "/home/akira/Dropbox/SpCoSLAM/learning/lang_m/"
#lang_init = 'web.000.htkdic' #'phonemes.htkdic' #  初期の単語辞書（./lang_mフォルダ内）

datasetfolder = "/home/akira/Dropbox/SpCoSLAM/rosbag/"
dataset1 = "albert-b-laser-vision/albert-B-laser-vision-dataset/"
bag1 = "albertBimg.bag"
dataset2 = "MIT_Stata_Center_Data_Set/"   ##用意できてない
#datasets = {"albert":dataset1,"MIT":dataset2}
datasets = [dataset1,dataset2]
bags = [bag1]
scantopic = ["scan", "base_scan _odom_frame:=odom_combined"]
#map_data : ./jygame/__inti__.py 

correct_Ct = 'Ct_correct.csv'  #データごとの正解のCt番号
correct_It = 'It_correct.csv'  #データごとの正解のIt番号
correct_data = 'SpCoSLAM_human.csv'  #データごとの正解の文章（単語列、区切り文字つき）(./data/)
correct_name = 'name_correct.csv'  #データごとの正解の場所の名前（音素列）

N_best_number = 10  #PRR評価用のN-bestのN
margin = 10*0.05 # 地図のグリッドと位置の値の関係が不明のため(0.05m/grid)*margin(grid)=0.05*margin(m)