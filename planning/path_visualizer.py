#coding:utf-8
#Akira Taniguchi 2019/01/22-
#For Visualization of Path
import sys
from math import pi as PI
from math import cos,sin,sqrt,exp,log,fabs,fsum,degrees,radians,atan2
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm
#import seaborn as sns
#import pandas as pd
from __init__ import *
from submodules import *

#マップを読み込む⇒確率値に変換⇒2次元配列に格納
def ReadMap(outputfile):
    #outputfolder + trialname + navigation_folder + map.csv
    gridmap = np.loadtxt(outputfile + "map.csv", delimiter=",")
    print "Read map: " + outputfile + "map.csv"
    return gridmap


########################################
if __name__ == '__main__': 
    #地図のファイルを読み込む

    #パスを読み込む

    #地図の上にパスを加える


    #地図をカラー画像として保存

