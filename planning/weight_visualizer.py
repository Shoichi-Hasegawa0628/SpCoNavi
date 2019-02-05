#coding:utf-8
#Akira Taniguchi 2019/01/22-2019/02/05-
#For Visualization of Posterior emission probability (PathWeightMap) on the grid map
import sys
#from math import pi as PI
#from math import cos,sin,sqrt,exp,log,fabs,fsum,degrees,radians,atan2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
#import seaborn as sns
#import pandas as pd
from __init__ import *
#from submodules import *
##実行コマンド例：
##python ./weight_visualizer.py alg2wicWSLAG10lln008 8


#マップを読み込む⇒確率値に変換⇒2次元配列に格納
def ReadMap(outputfile):
    #outputfolder + trialname + navigation_folder + map.csv
    gridmap = np.loadtxt(outputfile + "map.csv", delimiter=",")
    print "Read map: " + outputfile + "map.csv"
    return gridmap

#パス計算のために使用した確率値マップをファイル読み込みする
def ReadProbMap(outputfile):
    # 結果をファイル読み込み
    output = outputfile + "N"+str(N_best)+"G"+str(speech_num) + "_PathWeightMap.csv"
    PathWeightMap = np.loadtxt(output, delimiter=",")
    print "Read PathWeightMap: " + output
    return PathWeightMap

#ROSの地図座標系をPython内の2次元配列のインデックス番号に対応付ける
def Map_coordinates_To_Array_index(X):
    X = np.array(X)
    Index = np.round( (X - origin) / resolution ).astype(int) #四捨五入してint型にする
    return Index

#Python内の2次元配列のインデックス番号からROSの地図座標系への変換
def Array_index_To_Map_coordinates(Index):
    Index = np.array(Index)
    X = np.array( (Index * resolution) + origin )
    return X

#https://qiita.com/AnchorBlues/items/0dd1499196670fdf1c46
def draw(data,cb_min,cb_max):  #cb_min,cb_max:カラーバーの下端と上端の値
    X,Y=np.meshgrid(np.arange(data.shape[1]),np.arange(data.shape[0]))
    plt.figure(figsize=(10,4))  #図の縦横比を指定する
    div=20.0                    #図を描くのに何色用いるか
    delta=(cb_max-cb_min)/div
    interval=np.arange(cb_min,abs(cb_max)*2+delta,delta)[0:int(div)+1]
    plt.contourf(X,Y,data,interval)
    plt.show()
    

########################################
if __name__ == '__main__': 
    #学習済みパラメータフォルダ名を要求
    trialname = sys.argv[1]
    #print trialname
    #trialname = raw_input("trialname?(folder) >")

    #音声命令のファイル番号を要求   
    speech_num = sys.argv[2] #0
  
    ##FullPath of folder
    filename = datafolder + trialname + "/" + str(step) +"/"
    print filename #, particle_num
    outputfile = outputfolder + trialname + navigation_folder

    #地図のファイルを読み込む
    gridmap = ReadMap(outputfile)

    #PathWeightMapを読み込む
    PathWeightMap = ReadProbMap(outputfile)

    y_min = 380 #X_init_index[0] - T_horizon
    y_max = 800 #X_init_index[0] + T_horizon
    x_min = 180 #X_init_index[1] - T_horizon
    x_max = 510 #X_init_index[1] + T_horizon
    #if (x_min>=0 and x_max<=map_width and y_min>=0 and y_max<=map_length):
    PathWeightMap = PathWeightMap[x_min:x_max, y_min:y_max] # X[-T+I[0]:T+I[0],-T+I[1]:T+I[1]]
    gridmap = gridmap[x_min:x_max, y_min:y_max]

    #MAPの縦横(length and width)のセルの長さを計る
    map_length = len(PathWeightMap)  #len(costmap)
    map_width  = len(PathWeightMap[0])  #len(costmap[0])
    print "MAP[length][width]:",map_length,map_width

    #地図の上に重み(ヒートマップ形式)を加える
    plt.imshow(gridmap + (50+1)*(gridmap == -1), origin='lower', cmap='binary', vmin = 0, vmax = 100) #, vmin = 0.0, vmax = 1.0)
    #wmax = np.log(np.max(PathWeightMap))
    #wmin = np.log(np.min(PathWeightMap))
    plt.imshow(PathWeightMap,norm=LogNorm(), origin='lower', cmap='viridis') #, vmin=wmin, vmax=wmax) #gnuplot, inferno,magma,plasma  #
    #extent=[0, PathWeightMap.shape[1], PathWeightMap.shape[0],0 ] ) #, vmin = 0.0, vmax = 1.0) + np.log((np.exp(wmin)+np.exp(wmax))/2.0)
    

    pp=plt.colorbar (orientation="vertical",shrink=0.8) # カラーバーの表示 
    pp.set_label("Probability (log scale)", fontname="Arial", fontsize=10) #カラーバーのラベル
    pp.ax.tick_params(labelsize=8)
    plt.tick_params(axis='x', which='major', labelsize=8)
    plt.tick_params(axis='y', which='major', labelsize=8)
    #plt.xlim([380,800])             #x軸の範囲
    #plt.ylim([180,510])             #y軸の範囲
    plt.xlabel('X', fontsize=10)
    plt.ylabel('Y', fontsize=10)

    #地図をカラー画像として保存
    output = outputfile + "N"+str(N_best)+"G"+str(speech_num)
    plt.savefig(output + '_PathWeight.eps', dpi=300)#, transparent=True
    plt.savefig(output + '_PathWeight.png', dpi=300)#, transparent=True
    plt.savefig(output + '_PathWeight.pdf', dpi=300)#, transparent=True

    #plt.show()
    
    ########################################
    """
    p2 = PathWeighMap

    #y軸z軸作成
    yy,zz = [],[]
    y = p2[0,:]
    z = p2[:,0]
    y = y[1:]
    z = z[1:]

    for num in range(len(y)):
        yy.append(y)
    for num in range(len(z)):
        zz.append(z)
    Y = np.array(yy)
    Z = np.array(zz).T

    #データ2次元配列生成
    p2 = np.delete(p2,0,1)
    p2 = np.delete(p2,0,0)

    #描画
    plt.contourf(Y,Z,p2,100)
    plt.xlabel("y")
    plt.ylabel("z")
    plt.show()
    """

    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    H = ax.hist2d(x,y, bins=40)
    ax.set_title('1st graph')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(H[3],ax=ax)
    plt.show()
    """

    """
    min_index = [0,0]
    min_map = Array_index_To_Map_coordinates(min_index)
    max_index = [map_width,map_length]
    max_map = Array_index_To_Map_coordinates(max_index)
    
    x = np.arange(min_map[0], max_map[0], resolution) #x軸の描画範囲の生成。0から10まで0.05刻み。
    y = np.arange(min_map[1], max_map[1], resolution) #y軸の描画範囲の生成。0から10まで0.05刻み。

    X, Y = np.meshgrid(x, y)
    Z = PathWeightMap[x][y] #np.sin(X) + np.cos(Y)   # 表示する計算式の指定。等高線はZに対して作られる。

    plt.pcolormesh(X, Y, Z, cmap='hsv') # 等高線図の生成。cmapで色付けの規則を指定する。
    #plt.pcolor(X, Y, Z, cmap='hsv') # 等高線図の生成。cmapで色付けの規則を指定する。
    """

    #plt.title("100x100 matrix")
    #ax = plt.gca()
    #ax.invert_xaxis()
    #ax.invert_yaxis()



