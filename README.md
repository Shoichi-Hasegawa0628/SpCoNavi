# spconavi_ros
This repository is ros wrapper of SpCoNavi.  
Original SpCoNavi code is here： [https://github.com/a-taniguchi/SpCoNavi](https://github.com/a-taniguchi/SpCoNavi)

*   Maintainer: Shoichi Hasegawa ([hasegawa.shoichi@em.ci.ritsumei.ac.jp](mailto:hasegawa.shoichi@em.ci.ritsumei.ac.jp)).
*   Author: Shoichi Hasegawa ([hasegawa.shoichi@em.ci.ritsumei.ac.jp](mailto:hasegawa.shoichi@em.ci.ritsumei.ac.jp)).

## Content

*   [Execution Environment](#execution-environment)
*   [Learning Procedure](#learning-procedure)
*   [Planning Procedure](#planning-procedure)
*   [Folder](#folder)
*   [Codes of SpCoNavi](#codes-of-spconavi)
*   [Reference](#reference)
*   [Other Repositories](#other-repositories)
*   [Acknowledgements](#acknowledgements)


## Execution environment  
- Ubuntu：18.04LTS
- Python：2.7.17 (numpy：1.16.6, scipy：1.2.2, matplotlib：2.1.1)
- ROS：Melodic
- Robot：Turtlebot3 Waffle Pi


## Learning Procedure
`trialname` is the data folder name of the learning result in SpCoSLAM.  
For example, trialname is `3LDK_01` in `data` folder. 

### Command for acquiring costmap

~~~
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
roslaunch turtlebot3_slam turtlebot3_slam.launch slam_methods:=gmapping
python spconavi_costmap_generate.py trialname
rosrun map_server map_saver -f ../data/trialname/navi
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
### Command for test execution of SpCoNavi
Setting parameters and PATH in `__init__.py`  
~~~
roslaunch spconavi_ros spconavi_default.launch
~~~
Enter the number according to the algorithm used.

### Command for visualization of a path trajectory and the emission probability on the map
~~~
cd spconavi_ros/src/planning
python spconavi_output_map.py trialname init_position_num speech_num  
~~~
Example: 
`python spconavi_output_map.py 3LDK_01 0 7`  


## Folder  
 - `/src/`: source code folder
 - `/launch/`: launch folder
 - `/data/`: Data folder including sample data


## Codes of SpCoNavi
 - `README.md`: Read me file (This file)

 - `spconavi_ros_default.launch`: Execute spconavi launch file.

 - `spconavi_spatial_concepts_visualizer.launch`: Visualizer spatial concepts by launch file

 - `spconavi_execute.py`:  Execute code for SpCoNavi

 - `spconavi_viterbi_path_calculate.py`:  Execute code for SpCoNavi (Viterbi Algorithm)

 - `spconavi_astar_path_calculate.py`: Execute code for SpCoNavi (A* Algorithm)

 - `spconavi_read_data.py`: Module of reading pre-trained spatial concept model parameter, etc.

 - `spconavi_save_data.py`: Module of saving result of path, etc.

 - `spconavi_math.py`: Sub-program for functions

 - `spconavi_output_map.py`: Program for visualization of path trajectory and emission probability (log scale) 

 - `spconavi_costmap_generate.py`: Generate costmap

 - `spconavi_save_path_weight_map.py`: Save path weight map from learning data

 - `spconavi_path_visualizer_follower_rulo.py`: Visualize and follower path for rulo

 - `spconavi_path_visualizer_follower_hsr.py`: Visualize and follower path for hsr

 - `spconavi_spatial_concepts_visualizer.py`: Visualize learning spatial concepts

 - `__init__.py`: Code for initial setting (PATH and parameters)

 

## Reference
[1]: Akira Taniguchi, Yoshinobu Hagiwara, Tadahiro Taniguchi, and Tetsunari Inamura, "[Improved and scalable online learning of spatial concepts and language models with mapping](https://link.springer.com/article/10.1007/s10514-020-09905-0)", Autonomous Robots, Vol.44, pp927-pp946, 2020.  
[2]: Akira Taniguchi, Yoshinobu Hagiwara, Tadahiro Taniguchi, Tetsunari Inamura, "[Spatial concept-based navigation with human speech instructions via probabilistic inference on Bayesian generative model](https://www.tandfonline.com/doi/full/10.1080/01691864.2020.1817777)", Advanced Robotics, pp1213-pp1228, 2020.  


## Other repositories  
 - [SpCoSLAM_Lets](https://github.com/EmergentSystemLabStudent/SpCoSLAM_Lets): Wrapper of SpCoSLAM for mobile robots  
 - [SpCoSLAM](https://github.com/a-taniguchi/SpCoSLAM): Implementation of SpCoSLAM (Online Spatial Concept and Lexical Acquisition with Simultaneous Localization and Mapping)   
 - [SpCoSLAM 2.0](https://github.com/a-taniguchi/SpCoSLAM2): An Improved and Scalable Online Learning of Spatial Concepts and Language Models with Mapping (New version of online learning algorithm)   
 - [SpCoSLAM_evaluation](https://github.com/a-taniguchi/SpCoSLAM_evaluation): The codes for the evaluation or the visualization in our paper  
 - [SpCoNavi](https://github.com/a-taniguchi/SpCoNavi): Spatial Concept-Based Navigation from Human Speech Instructions by Probabilistic Inference on Bayesian Generative Model