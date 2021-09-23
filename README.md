# spconavi_ros
This repository is ros wrapper of SpCoNavi.  
Original SpCoNavi code is here： [https://github.com/a-taniguchi/SpCoNavi](https://github.com/a-taniguchi/SpCoNavi)

*   Maintainer: Shoichi Hasegawa ([hasegawa.shoichi@em.ci.ritsumei.ac.jp](mailto:hasegawa.shoichi@em.ci.ritsumei.ac.jp)).
*   Author: Shoichi Hasegawa ([hasegawa.shoichi@em.ci.ritsumei.ac.jp](mailto:hasegawa.shoichi@em.ci.ritsumei.ac.jp)).

## Content

*   [Execution Environment](#execution-environment)
*   [Preparation](#preparation)
*   [Execute Procedure](#execute-procedure)
*   [Folder](#folder)
*   [Codes of SpCoNavi](#codes-of-spconavi)
*   [Reference](#reference)
*   [Acknowledgements](#acknowledgements)


## Execution environment  
- Ubuntu：18.04LTS
- Python：2.7.17 (numpy：1.16.6, scipy：1.2.2, matplotlib：2.1.1)
- ROS：Melodic
- Robot：Turtlebot3 Waffle Pi


## Preparation
`trialname` is the data folder name of the learning result in SpCoSLAM.  
For example, trialname is `3LDK_01` in `data` folder. 

### Command for acquiring costmap
~~~
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
roslaunch turtlebot3_slam turtlebot3_slam.launch slam_methods:=gmapping
python generate_costmap.py trialname
rosrun map_server map_saver -f ../data/trialname/navi
~~~

### Command for acquiring path_weight_map
~~~
python generate_path_weight_map.py
~~~

### Command for learning of spatial concepts  
Learn the concept of place with SpCoSLAM or SpCoA.  
The following is a reference link.  
 - [SpCoSLAM](https://github.com/a-taniguchi/SpCoSLAM): Implementation of SpCoSLAM (Online Spatial Concept and Lexical Acquisition with Simultaneous Localization and Mapping)   
 - [SpCoSLAM 2.0](https://github.com/a-taniguchi/SpCoSLAM2): An Improved and Scalable Online Learning of Spatial Concepts and Language Models with Mapping (New version of online learning algorithm)   
After learning, store it in the data folder so that it matches the path.  

### Visulalization of the learning result  
~~~
roslaunch spconavi_ros spconavi_spatial_concepts_visualizer.launch
~~~

## Planning Procedure
Setting parameters and PATH in `__init__.py`  
~~~
roslaunch spconavi_ros spconavi_ros_default.launch
~~~
Enter the number according to the algorithm used.


## Folder  
 - `/scripts/`: source code folder
 - `/launch/`: launch folder
 - `/data/`: Data folder including sample data
 - `/configs/`: parameter folder


## Codes of SpCoNavi
 - `README.md`: Read me file (This file)

 - `spconavi_ros_default.launch`: Execute spconavi launch file.

 - `spconavi_spatial_concepts_visualizer.launch`: Visualizer spatial concepts by launch file

 - `execute_node.py`:  Execute code for SpCoNavi

 - `viterbi_path_calculate.py`:  Execute code for SpCoNavi (Viterbi Algorithm)

 - `astar_path_calculate.py`: Execute code for SpCoNavi (A* Algorithm)

 - `dataset.py`: Module of reading pre-trained spatial concept model parameter, etc.

 - `converter.py`: Module of converting data, etc.

 - `spconavi_math.py`: Sub-program for functions

 - `generate_costmap.py`: Generate costmap

 - `generate_path_weight_map.py`: Generate path_weight_map

 - `path_visualizer_follower_rulo.py`: Visualize and follower path for rulo

 - `path_visualizer_follower_hsr.py`: Visualize and follower path for hsr

 - `spatial_concepts_visualizer.py`: Visualize learning spatial concepts

 - `__init__.py`: Code for initial setting (PATH and parameters)

 

## Reference
[1]: Akira Taniguchi, Yoshinobu Hagiwara, Tadahiro Taniguchi, and Tetsunari Inamura, "[Improved and scalable online learning of spatial concepts and language models with mapping](https://link.springer.com/article/10.1007/s10514-020-09905-0)", Autonomous Robots, Vol.44, pp927-pp946, 2020.  
[2]: Akira Taniguchi, Yoshinobu Hagiwara, Tadahiro Taniguchi, Tetsunari Inamura, "[Spatial concept-based navigation with human speech instructions via probabilistic inference on Bayesian generative model](https://www.tandfonline.com/doi/full/10.1080/01691864.2020.1817777)", Advanced Robotics, pp1213-pp1228, 2020.  

