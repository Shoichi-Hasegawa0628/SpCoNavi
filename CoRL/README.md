# SpCoNavi for SIGVerse
SpCoNavi: Spatial Concept-Based Navigation from Human Speech Instructions by Probabilistic Inference on Bayesian Generative Model  
<img src="https://github.com/a-taniguchi/SpCoNavi/blob/master/img/outline.png" width="480px">


## Execution environment  
- Ubuntu 16.04 (on VMware)  
    - Python 2.7.12  
    - ROS kinetic  
- Windows 10  
    - Unity 2018.4.0f1  

## Preparation for execution (変更中…)  
### Windows  
SIGVerseのサイトに従い、UnityとSIGVerseProjectをセットアップ  
サンプルのHSRのHsrTeleopを起動  

【部屋環境の変更】  
（予定）  


### Ubuntu  
【準備コマンド】  
~~~
sudo pip install numpy scipy matplotlib numba
~~~

【Notes】  
実行時に、以下のようなWarning文が出るときは、pipでcoloramaをインストールorアップデートしてみてください。  
~~~
/usr/local/lib/python2.7/dist-packages/numba/errors.py:104: UserWarning: Insufficiently recent colorama version found. Numba requires colorama >= 0.3.9
  warnings.warn(msg)
~~~

~~~
sudo pip install colorama
~~~

## Execution procedure
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
'alg2wicWSLAG10lln008'の箇所には、SpCoSLAM実行時の'trialname'を指定してください。  

【SpCoNaviのテスト実行コマンド】  
~~~
python ./SpCoNavi0.1.py trialname particle_num init_position_num speech_num  
~~~
例：
~~~
python ./SpCoNavi0.1.py alg2wicWSLAG10lln008 0 0 0
~~~

【出力確率とパスの可視化の実行コマンド】
~~~
python ./path\_weight\_visualizer.py trialname speech_num  
~~~
例：
~~~
python ./path_weight_visualizer.py alg2wicWSLAG10lln008 8
~~~

## Folder  
フォルダ構成は後ほど変更予定  
 - `/costmap_global/`: Get 2d costmap
 - `/data/alg2wicWSLAG10lln008/navi/`: Sample output data
 - `/img/`: Image files for README.md
 - `/planning/`: Main codes for planning
 
---
## Reference
[1]: Akira Taniguchi, Yoshinobu Hagiwara, Tadahiro Taniguchi, and Tetsunari Inamura, "Online Spatial Concept and Lexical Acquisition with Simultaneous Localization and Mapping", IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2017.  
[2]: 谷口彰，萩原良信，谷口忠大，稲邑哲也. 場所概念に基づく確率推論による音声命令からのパスプランニング. 人工知能学会全国大会 (JSAI). 2019. (Japanese)  

【Other repositories】  
 - [SpCoSLAM_Lets](https://github.com/EmergentSystemLabStudent/SpCoSLAM_Lets): Wrapper of SpCoSLAM for mobile robots (Recommended)  
 - [SpCoSLAM](https://github.com/a-taniguchi/SpCoSLAM): Implementation of SpCoSLAM (Online Spatial Concept and Lexical Acquisition with Simultaneous Localization and Mapping)   
 - [SpCoSLAM 2.0](https://github.com/a-taniguchi/SpCoSLAM2): An Improved and Scalable Online Learning of Spatial Concepts and Language Models with Mapping (New version of online learning algorithm)   
 - [SpCoSLAM_evaluation](https://github.com/a-taniguchi/SpCoSLAM_evaluation): The codes for the evaluation or the visualization in our paper  

2019/06/25  Akira Taniguchi  

