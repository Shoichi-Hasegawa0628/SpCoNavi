#! /usr/bin/env python
#coding:utf-8
import numpy as np

#################### Definition of Place name ########################################################################

# Note: Don't be tupple! Only list! [*,*]
# 原点は[[200, 200]], 玄関は[[264, 250]]. (長谷川RSJ2021で使用した環境)
# 計算方法は「SpCoNavi上の(y, x)座標(pixel) * 0.05(m/pixel) +(-10, -10) (m) = Gazebo上の座標(m)」
# [192, 192] is (y,x). not (x,y). (Same as coordinates in Astar_*.py) 

Start_Position = [[264, 250]] 
Goal_Word = ["living", "kitchen", "bedroom", "toilet"]
Example_AND = ["北","寝室"] 
Example_OR = ["ダイニング","キッチン"] 

#################### Folder PATH ####################################################################################

# Data Path
inputfolder_SIG = "/root/RULO/catkin_ws/src/spco2_mlda_problog/spconavi_ros/data/"
outputfolder_SIG = "/root/RULO/catkin_ws/src/spco2_mlda_problog/spconavi_ros/data/"

#Navigation folder (Other output files are also in same folder.)
navigation_folder = "/navi/"  #outputfolder + trialname + / + navigation_folder + contmap.csv

#Cost map folder
costmap_folder = navigation_folder  #"/costmap/" 

#################### Parameters ###################################################################################

#Same values as /learning/__init.py__
L = 4 #10 #100                  #The number of spatial concepts
K = 4 #10 #100                  #The number of position distributions
D = 3                           # the number of object categories

memory_reduction = 1 #0 #Memory reduction process (ON:1, OFF:0)
NANAME = 0              #Action pattern: up, down, left and right (0), and add diagonal (oblique) movements (１)
word_increment = 6 #10     #Increment number of word observation data (BoWs)

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

UPDATE_PostProbMap = 1 #0 #If the file exists already, calculate PostProbMap: (1) 

#Select approximated methods (Proposed method (ver. SIGVerse):0) -> run SpCoNavi_Astar_approx.py
Approx = 0  
if (NANAME != 1):
  Approx = 1
#Separated N-best approximation version is another program (SpCoNavi0.1s.py)

# SpCoNavi_Astar_approx.py: The number of goal position candidates
Sampling_J = 10 #10

#Dynamics of state transition (motion model): (Deterministic:0, Probabilistic:1, Approximation:2(Unimplemented))
#Dynamics = 0

cmd_vel = 1  #Movement amount of robot (ROS: cmd_vel [m/s], [rad/s]) [default:1 (int)]
#MotionModelDist = "Gauss"  #"Gauss": Gaussian distribution, "Triangular": Triangular distribution

#Odometry motion model parameters (Same values to AMCL or gmapping): unused
#odom_alpha1 = 0.2  #(double, default: 0.2) Specifies the expected noise in odometry's rotation estimate from the rotational component of the robot's motion. 
#odom_alpha2 = 0.2  #(double, default: 0.2) Specifies the expected noise in odometry's rotation estimate from translational component of the robot's motion. 
#odom_alpha3 = 0.2  #(double, default: 0.2) Specifies the expected noise in odometry's translation estimate from the translational component of the robot's motion. 
#odom_alpha4 = 0.2  #(double, default: 0.2) Specifies the expected noise in odometry's translation estimate from the rotational component of the robot's motion. 

#################### Others #####################################################################################

#ROS topic name
MAP_TOPIC     = "/map"
COSTMAP_TOPIC = "/move_base/global_costmap/costmap"

#Same value to map yaml file
resolution = 0.050000
origin     = np.array([-10.000000, -10.000000]) #np.array([x,y])

#dimx = 2           #The number of dimensions of xt (x,y)
#margin = 10*0.05   #margin value for place area in gird map (0.05m/grid)*margin(grid)=0.05*margin(m)
approx_log_zero = np.log(10.0**(-300))   #approximated value of log(0)

