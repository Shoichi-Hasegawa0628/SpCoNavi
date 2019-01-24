# SpCoNavi
SpCoNavi: Spatial Concept-Based Navigation from Human Speech Instructions by Probabilistic Inference on Bayesian Generative Model

## Abstract
TBA  

Figure1: The graphical model of SpCoSLAM: Online Spatial Concept and Lexical Acquisition with Simultaneous Localization and Mapping   
<img src="https://github.com/a-taniguchi/SpCoSLAM/blob/master/img/graphicalmodel02.jpg" width="520px">

Figure2: The graphical model of SpCoNavi: Spatial Concept-based Path-Planning   
<img src="https://github.com/a-taniguchi/SpCoNavi/blob/master/img/gmodel_spconavi_simple2.png" width="520px">

## Execution environment  
- Ubuntu 14.04  
- Python 2.7.6  
- ROS indigo  
- Speech recognition system: Julius dictation-kit-v4.4 GMM-HMM/DNN-HMM (Using Japanese syllabary dictionary, N-best output)  

## Preparation for execution  
TBA  

## Execution procedure
TBA  

【コストマップ取得のためのコマンドリスト】  
~~~
(別ターミナル：起動しなくてもいい)
roscore
---
(別ターミナル：環境設定は代替可能のはず)
roslaunch turtlebot_gazebo turtlebot_world.launch
---
(別ターミナル)
source ~/Dropbox/SpCoNavi/costmap_global/devel/setup.bash
roslaunch fourth_robot_2dnav global_costmap.launch

（mapファイル指定する場合）
roslaunch fourth_robot_2dnav global_costmap.launch map_file:=my_map.yaml
---
(別ターミナル)
cd ~/Dropbox/SpCoNavi/planning
python costmap.py alg2wicWSLAG10lln008
~~~

【SpCoNaviのテスト実行コマンド】  
~~~
python ./SpCoNavi0.01.py alg2wicWSLAG10lln008 0 0 0
~~~

【準備コマンド】  
~~~
sudo pip install numba
sudo pip install colorama
~~~

## Notes
TBA  

---
【Other repositories】  
 [SpCoSLAM_Lets](https://github.com/EmergentSystemLabStudent/SpCoSLAM_Lets): Wrapper of SpCoSLAM for mobile robots (Recommended)  
 [SpCoSLAM](https://github.com/a-taniguchi/SpCoSLAM): Implementation of SpCoSLAM (Online Spatial Concept and Lexical Acquisition with Simultaneous Localization and Mapping)   
 [SpCoSLAM 2.0](https://github.com/a-taniguchi/SpCoSLAM2): An Improved and Scalable Online Learning of Spatial Concepts and Language Models with Mapping (New version of online learning algorithm)   
 [SpCoSLAM_evaluation](https://github.com/a-taniguchi/SpCoSLAM_evaluation): The codes for the evaluation or the visualization in our paper  

2019/01/22  Akira Taniguchi  
