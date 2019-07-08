# SpCoNavi for SIGVerse
SpCoNavi: Spatial Concept-Based Navigation from Human Speech Instructions by Probabilistic Inference on Bayesian Generative Model  
<img src="https://github.com/a-taniguchi/SpCoNavi/blob/master/img/outline.png" width="480px">


## Execution environment  
- Ubuntu 16.04 (on VMware)  
    - Python 2.7.12 and 3.5.2  
    - ROS kinetic  
- Windows 10  
    - Unity 2018.4.0f1  

## Preparation for execution (変更中…)  
### Windows  
SIGVerseのサイトに従い、UnityとSIGVerseProjectをセットアップ  
サンプルのHSRのHsrTeleopを起動  

【部屋環境の変更】  
(TBA)  


### Ubuntu  
【Command for preparation】  
~~~
sudo pip install numpy scipy matplotlib
~~~

SIGVerseのサイトに従い、セットアップ  
ROS Tutorialを順番に実行  


## Execution procedure
【Command list for cost map acquisition】  
~~~
(Terminal: There is no problem if this is not done)
roscore
---
(Another terminal: This is an example of an execution command. Environment settings can be replaced with others.)
roslaunch turtlebot_gazebo turtlebot_world.launch
---
(Another terminal)
source ~/*/SpCoNavi/costmap_global/devel/setup.bash
roslaunch fourth_robot_2dnav global_costmap.launch

（If you specify a map yaml file.）
roslaunch fourth_robot_2dnav global_costmap.launch map_file:=my_map.yaml
---
(Another terminal)
cd ~/*/SpCoNavi/planning
python costmap.py trialname
~~~
'trialname' is the data folder name of the learning result in SpCoSLAM.  
For example, trialname is 'alg2wicWSLAG10lln008' in 'data' folder.  


Command for test execution of SpCoNavi】  
~~~
python ./SpCoNavi0.1.py trialname particle_num init_position_num speech_num  
~~~
Example：
~~~
python ./SpCoNavi0.1.py alg2wicWSLAG10lln008 0 0 0
~~~

【Command for visualization of a path trajectory and the emission probability on the map】
~~~
python ./path_weight_visualizer.py trialname speech_num  
~~~
Example：
~~~
python ./path_weight_visualizer.py alg2wicWSLAG10lln008 8
~~~

## Folder  
フォルダ構成は後ほど変更予定  
 - `/planning/`: Main codes for planning
 - TBA
 
---
## Reference
[1]: Akira Taniguchi, Yoshinobu Hagiwara, Tadahiro Taniguchi, and Tetsunari Inamura, "Online Spatial Concept and Lexical Acquisition with Simultaneous Localization and Mapping", IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2017.  
[2]: Akira Taniguchi, Yoshinobu Hagiwara, Tadahiro Taniguchi, Tetsunari Inamura, "Path Planning by Spatial Concept-Based Probabilistic Inference from Human Speech Instructions", the 33rd Annual Conference of the Japanese Society for Artificial Intelligence, 2019. (In Japanese; 谷口彰，萩原良信，谷口忠大，稲邑哲也. 場所概念に基づく確率推論による音声命令からのパスプランニング. 人工知能学会全国大会 (JSAI). 2019.)    


【Other repositories】  
 - [SpCoSLAM_Lets](https://github.com/EmergentSystemLabStudent/SpCoSLAM_Lets): Wrapper of SpCoSLAM for mobile robots (Recommended)  
 - [SpCoSLAM](https://github.com/a-taniguchi/SpCoSLAM): Implementation of SpCoSLAM (Online Spatial Concept and Lexical Acquisition with Simultaneous Localization and Mapping)   
 - [SpCoSLAM 2.0](https://github.com/a-taniguchi/SpCoSLAM2): An Improved and Scalable Online Learning of Spatial Concepts and Language Models with Mapping (New version of online learning algorithm)   
 - [SpCoSLAM_evaluation](https://github.com/a-taniguchi/SpCoSLAM_evaluation): The codes for the evaluation or the visualization in our paper  

2019/06/25  Akira Taniguchi  
2019/07/08  Akira Taniguchi (Update)  
