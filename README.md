# spconavi_ros
SpCoNavi: Spatial Concept-Based Navigation from Human Speech Instructions by Probabilistic Inference on Bayesian Generative Model  
<img src="https://github.com/a-taniguchi/SpCoNavi/blob/master/img/outline.png" width="480px">

## Abstract
The aim of this study is to enable a mobile robot to perform navigation tasks by probabilistic inference using spatial concepts on a probabilistic model. 
Specifically, path planning is performed to the target state of a spatial concept estimated through human speech instructions such as ``Go to the kitchen''.
In the experiment, places instructed by the speech command of the user showed high probability values, and the trajectory toward the target place was correctly estimated.   

| Fig.1: The graphical model of SpCoSLAM: Online Spatial Concept and Lexical Acquisition with Simultaneous Localization and Mapping [1] | Fig.2: The graphical model of SpCoNavi: Spatial Concept-based Path-Planning [2,3] | 
| :---: | :---: | 
| <img src="https://github.com/a-taniguchi/SpCoSLAM/blob/master/img/graphicalmodel02.jpg" width="320px"> | <img src="https://github.com/a-taniguchi/SpCoNavi/blob/master/img/gmodel_spconavi_simple2.png" width="320px"> | 


## Execution environment  
- Ubuntu 14.04  
- Python 2.7.6  
- ROS indigo  
- Speech recognition system: Julius dictation-kit-v4.4 GMM-HMM/DNN-HMM (N-best output)  

## Preparation for execution  
【Spatial concept learning】  
SpCoNavi needs the learning data files by SpCoSLAM or other spatial concept formation methods.  
It is assumed that a reasonably accurate map of the environment, spatial concepts, and lexicon, will have already been acquired. 
This assumption implies that SpCoSLAM has already been in operation in the environment and each model parameter for the learning result from the last time-step has been fixed.   

【Command for preparation】  
~~~
sudo pip install numpy scipy matplotlib numba
~~~

Notes: If you get the following warning statement at runtime, please install or update `colorama` with pip.  
~~~
/usr/local/lib/python2.7/dist-packages/numba/errors.py:104: UserWarning: Insufficiently recent colorama version found. Numba requires colorama >= 0.3.9
  warnings.warn(msg)
~~~

~~~
sudo pip install colorama
~~~

If you get an error about `numba`, comment out the relevant part about `numba` in the navigation code and use it.  

## Execution procedure
【Command list for cost map acquisition】  
~~~
(Terminal: There is no problem if this is not done)
roscore
---
(Another terminal: This is an example of an execution command. Environment settings can be replaced with others.)
roslaunch turtlebot_gazebo turtlebot_world.launch
---
(Another terminal: It needs catkin_make in the /costmap_global/ folder before running the following commands.)
source ~/*/SpCoNavi/costmap_global/devel/setup.bash
roslaunch fourth_robot_2dnav global_costmap.launch

(If you specify a map yaml file.)
roslaunch fourth_robot_2dnav global_costmap.launch map_file:=my_map.yaml
---
(Another terminal)
cd ~/*/SpCoNavi/planning
python costmap.py trialname
~~~
`trialname` is the data folder name of the learning result in SpCoSLAM.  
For example, trialname is `alg2wicWSLAG10lln008` in `data` folder.  


【Command for test execution of SpCoNavi】  
~~~
python ./SpCoNavi0.1s.py trialname particle_num init_position_num speech_num  
~~~
Example: 
`python ./SpCoNavi0.1s.py alg2wicWSLAG10lln008 0 0 0`  


【Command for visualization of a path trajectory and the emission probability on the map】
~~~
python ./path_weight_visualizer.py trialname speech_num  
~~~
Example: 
`python ./path_weight_visualizer.py alg2wicWSLAG10lln008 8`  


## Folder
 - `/costmap_global/`: To get 2d costmap
 - `/data/alg2wicWSLAG10lln008/navi/`: Sample output data
 - `/img/`: Image files for README.md
 - `/planning/`: Main codes for planning
 - `/SIGVerse/`: SpCoNavi for SIGVerse simulator environment
 
---
## Reference
[1]: Akira Taniguchi, Yoshinobu Hagiwara, Tadahiro Taniguchi, and Tetsunari Inamura, "Online Spatial Concept and Lexical Acquisition with Simultaneous Localization and Mapping", IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2017.  
[2]: Akira Taniguchi, Yoshinobu Hagiwara, Tadahiro Taniguchi, Tetsunari Inamura, "Path Planning by Spatial Concept-Based Probabilistic Inference from Human Speech Instructions", the 33rd Annual Conference of the Japanese Society for Artificial Intelligence, 2019. (In Japanese; 谷口彰，萩原良信，谷口忠大，稲邑哲也. 場所概念に基づく確率推論による音声命令からのパスプランニング. 人工知能学会全国大会 (JSAI). 2019.)    
[3]: Akira Taniguchi, Yoshinobu Hagiwara, Tadahiro Taniguchi, Tetsunari Inamura, "Spatial Concept-Based Navigation with Human Speech Instructions via Probabilistic Inference on Bayesian Generative Model", [arXiv:2002.07381](https://arxiv.org/abs/2002.07381).

## Other repositories  
 - [SpCoSLAM_Lets](https://github.com/EmergentSystemLabStudent/SpCoSLAM_Lets): Wrapper of SpCoSLAM for mobile robots (Recommended)  
 - [SpCoSLAM](https://github.com/a-taniguchi/SpCoSLAM): Implementation of SpCoSLAM (Online Spatial Concept and Lexical Acquisition with Simultaneous Localization and Mapping)   
 - [SpCoSLAM 2.0](https://github.com/a-taniguchi/SpCoSLAM2): An Improved and Scalable Online Learning of Spatial Concepts and Language Models with Mapping (New version of online learning algorithm)   
 - [SpCoSLAM_evaluation](https://github.com/a-taniguchi/SpCoSLAM_evaluation): The codes for the evaluation or the visualization in our paper  

2019/02/21  Akira Taniguchi  
2019/06/17  Akira Taniguchi (Update)  
2019/07/08  Akira Taniguchi (Update)  
