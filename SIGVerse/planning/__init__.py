#coding:utf-8
#The file for setting parameters
#Akira Taniguchi 2018/12/13-2019/03/10-2019/07/13
import numpy as np

##Command
#python ./SpCoNavi0.1_SIGVerse.py trialname iteration sample init_position_num speech_num
#python ./SpCoNavi0.1_SIGVerse.py 3LDK_01 1 0 0 0

##### NEW #####
inputfolder_SIG  = "/mnt/hgfs/Dropbox/SpCoNavi/CoRL/dataset/similar/3LDK/"  #"/home/akira/Dropbox/SpCoNavi/data/"
outputfolder_SIG = "/mnt/hgfs/Dropbox/SpCoNavi/CoRL/data/"  #"/home/akira/Dropbox/SpCoNavi/data/"

Start_Position = [[100,100],[100,110],[120,60],[60,90],[90,120],[75,75]] #(y,x). not (x,y). (Same as coordinates in Astar_*.py) 
Goal_Word = ["玄関","リビング","ダイニング","キッチン","風呂","洗面所","トイレ","寝室"] # In Japanese
#Goal_Word = ["Entrance","Living room","Dining room","Kitchen","Bath room","Washroom","Toilet","Bedroom"]
#0:玄関,1:リビング,2:ダイニング,3:キッチン,4:風呂,5:洗面所,6:トイレ,7:寝室,

#Same values as /learning/__init.py__
L = 10 #100                  #The number of spatial concepts
K = 10 #100                  #The number of position distributions

memory_reduction = 1 #0 #Memory reduction process (ON:1, OFF:0)
NANAME = 0              #Action pattern: up, down, left and right (0), and add diagonal (oblique) movements (１)
word_increment = 10     #Increment number of word observation data (BoWs)

#################### Folder PATH ####################
#Setting of PATH for a folder of learned spatial concept parameters
datafolder    = "/mnt/hgfs/D/Dropbox/SpCoSLAM/data/" #"/home/akira/Dropbox/SpCoSLAM/data/" 
#Setting of PATH for output folder
outputfolder  = "/mnt/hgfs/D/Dropbox/SpCoSLAM/data/"  #"/home/akira/Dropbox/SpCoNavi/data/"

#File folder of speech data
#speech_folder    = "/home/akira/Dropbox/Julius/directory/SpCoSLAM/*.wav"    #Teaching speech data folder
#speech_folder_go = "/home/akira/Dropbox/Julius/directory/SpCoSLAMgo/*.wav"  #Evaluation speech data folder
#lmfolder         = "/mnt/hgfs/D/Dropbox/SpCoSLAM/learning/lang_m/"  #Language model (word dictionary)

#Navigation folder (Other output files are also same folder)
navigation_folder = "/navi/"  #outputfolder + trialname + / + navigation_folder + contmap.csv
# follow folder format of learning result in spatial concept

#Cost map folder
costmap_folder = navigation_folder  #"/costmap/" 



#################### Parameters ####################
T_horizon  = 200     #Planning horizon #may be over 150~200. depends on memory and computational limits
N_best     = word_increment #10      #N of N-best (N<=10)
#step       = 50      #The end number of time-step in SpCoSLAM (the number of training data)

#Initial position (position candidates)
X_candidates = Start_Position  #Index coordinates on 2 dimension list

#When starting from the mid-flow (value of t to read trellis, from the beginning: 0)
T_restart = 0         #If T_horizon changes, it can not be used at present because the number of states changes in the dimension reduction process. If you do not save trellis, you run from the beginning.

SAVE_time    = 1      #Save computational time (Save:1, Not save:0)
SAVE_X_init  = 1      #Save initial value (Save:1, Not save:0) 
SAVE_T_temp  = 10     #Step interval to save the path temporarily (each SAVE_T_temp value on the way)
SAVE_Trellis = 0      #Save trellis for Viterbi Path estimation (Save:1, Not save:0) 

UPDATE_PostProbMap = 0 #1 #If the file exists already, calculate PostProbMap: (1) 

#Select approximated methods (Proposed method (ver. SIGVerse):0)
Approx = 0  
if (NANAME != 1):
  Approx = 1
#Separated N-best approximation version is another program (SpCoNavi0.1s.py)

#Dynamics of state transition (motion model): (Deterministic:0, Probabilistic:1, Approximation:2(Unimplemented))
#Dynamics = 0

cmd_vel = 1  #Movement amount of robot (ROS: cmd_vel [m/s], [rad/s]) [default:1 (int)]
#MotionModelDist = "Gauss"  #"Gauss": Gaussian distribution, "Triangular": Triangular distribution

#Odometry motion model parameters (Same values to AMCL or gmapping): unused
#odom_alpha1 = 0.2  #(double, default: 0.2) ロボットの動きの回転移動からオドメトリの回転移動のノイズ
#odom_alpha2 = 0.2  #(double, default: 0.2) ロボットの動きの平行移動からオドメトリの回転移動のノイズ
#odom_alpha3 = 0.2  #(double, default: 0.2) ロボットの動きの平行移動からオドメトリの平行移動のノイズ
#odom_alpha4 = 0.2  #(double, default: 0.2) ロボットの動きの回転移動からオドメトリの平行移動のノイズ
#srr = 0.1 #(float, default: 0.1) #オドメトリの誤差．平行移動に起因する平行移動の誤差．
#srt = 0.2 #(float, default: 0.2) #オドメトリの誤差．回転移動に起因する平行移動の誤差．
#str = 0.1 #(float, default: 0.1) #オドメトリの誤差．平行移動に起因する回転移動の誤差．
#stt = 0.2 #(float, default: 0.2) #オドメトリの誤差．回転移動に起因する回転移動の誤差．

#ROS topic name
MAP_TOPIC     = "/map"
COSTMAP_TOPIC = "/move_base/global_costmap/costmap"
#PATH_TOPIC = "/spconavi/path" #Unimplemented

#Same value to map yaml file
resolution = 0.1   #0.050000
origin =  np.array([-10.000000, -10.000000]) #np.array([x,y]) #np.array([-30.000000, -20.000000])

#map size (length and width)
#map_length = 0
#map_width  = 0

"""
#Julius parameters
JuliusVer      = "v4.4"   #"v.4.3.1"
HMMtype        = "DNN"    #"GMM"
lattice_weight = "AMavg"  #"exp" #acoustic likelihood (log likelihood: "AMavg", likelihood: "exp")
wight_scale    = -1.0
#WDs = "0"   #DNN版の単語辞書の音素を*_Sだけにする("S"), BIE or Sにする("S"以外)
##In other parameters, please see "main.jconf" in Julius folder

if (JuliusVer ==  "v4.4"):
  Juliusfolder = "/home/akira/Dropbox/Julius/dictation-kit-v4.4/"
else:
  Juliusfolder = "/home/akira/Dropbox/Julius/dictation-kit-v4.3.1-linux/"

if (HMMtype == "DNN"):
  lang_init = 'syllableDNN.htkdic' 
else:
  lang_init = 'syllableGMM.htkdic' 
  # 'trueword_syllable.htkdic' #'phonemes.htkdic' # 初期の単語辞書(./lang_mフォルダ内)
"""

#dimx = 2           #The number of dimensions of xt (x,y)
#margin = 10*0.05   #地図のグリッドと位置の値の関係が不明のため(0.05m/grid)*margin(grid)=0.05*margin(m)
approx_log_zero = np.log(10.0**(-300))   #approximated value of log(0)


"""
####################Particle Class (structure)####################
class Particle:
  def __init__(self,id,x,y,theta,weight,pid):
    self.id = id
    self.x = x
    self.y = y
    self.theta = theta
    self.weight = weight
    self.pid = pid
    #self.Ct = -1
    #self.it = -1

####################Option setting (NOT USE)####################
wic = 1         #1:wic重みつき(理論的にはこちらがより正しい), 0:wic重みなし(Orignal paper of SpCoSLAM)
UseFT = 1       #画像特徴を使う場合(１), 使わない場合(０)
UseLM = 1       #言語モデルを更新する場合(１), しない場合(０)[Without update language modelのため無関係]

#NbestNum = N_best      #N of N-best (N<=10)
#N_best_number = N_best #N of N-best (N<=10) for PRR

##Initial (hyper) parameters
##Posterior (∝likelihood×prior): https://en.wikipedia.org/wiki/Conjugate_prior
alpha0 = 10.0        #Hyperparameter of CRP in multinomial distribution for index of spatial concept
gamma0 = 1.0         #Hyperparameter of CRP in multinomial distribution for index of position distribution
beta0 = 0.1          #Hyperparameter in multinomial distribution P(W) for place names 
chi0  = 0.1          #Hyperparameter in multinomial distribution P(φ) for image feature
k0 = 1e-3            #Hyperparameter in Gaussina distribution P(μ) (Influence degree of prior distribution of μ)
m0 = np.zeros(dimx)  #Hyperparameter in Gaussina distribution P(μ) (prior mean vector)
V0 = np.eye(dimx)*2  #Hyperparameter in Inverse Wishart distribution P(Σ)(prior covariance matrix)
n0 = 3.0             #Hyperparameter in Inverse Wishart distribution P(Σ) {>the number of dimenssions] (Influence degree of prior distribution of Σ)
k0m0m0 = k0*np.dot(np.array([m0]).T,np.array([m0]))
"""
