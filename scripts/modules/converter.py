#!/usr/bin/env python
#coding:utf-8

# Third Party
from scipy.stats import multivariate_normal

# Self-made Modules
from __init__ import *

class Converter():
    
    #ROSのmap 座標系をPython内の2-dimension array index 番号に対応付ける
    def Map_coordinates_To_Array_index(self, X):
        X = np.array(X)
        Index = np.round( (X - origin) / resolution ).astype(int) #四捨五入してint型にする
        return Index


    #Python内の2-dimension array index 番号からROSのmap 座標系への変換
    def Array_index_To_Map_coordinates(self, Index):
        Index = np.array(Index)
        X = np.array( (Index * resolution) + origin )
        return X


    #gridmap and costmap から確率の形のCostMapProbを得ておく
    def CostMapProb_jit(self, gridmap, costmap):
        CostMapProb = (100.0 - costmap) / 100.0     #Change the costmap to the probabilistic costmap
        #gridの数値が0（非占有）のところだけ数値を持つようにマスクする
        GridMapProb = 1*(gridmap == 0)  #gridmap * (gridmap != 100) * (gridmap != -1)  #gridmap[][]が障害物(100)または未探索(-1)であれば確率0にする
        return CostMapProb * GridMapProb


    def PostProb_ij(self, Index_temp,Mu,Sig,Phi_l,LookupTable_ProbCt,map_length,map_width,L,K, CostMapProb):
        if (CostMapProb[Index_temp[1]][Index_temp[0]] != 0.0): 
            X_temp = self.Array_index_To_Map_coordinates(Index_temp)  #map と縦横の座標系の軸が合っているか要確認
            #print X_temp,Mu
            sum_i_GaussMulti = [ np.sum([multivariate_normal.pdf(X_temp, mean=Mu[k], cov=Sig[k]) * Phi_l[c][k] for k in xrange(K)]) for c in xrange(L) ] ##########np.array( ) !!! np.arrayにすると, numbaがエラーを吐く
            PostProb = np.sum( LookupTable_ProbCt * sum_i_GaussMulti ) #sum_c_ProbCtsum_i
        else:
            PostProb = 0.0
        return PostProb


    #@jit(parallel=True)  #並列化されていない？1CPUだけ使用される
    def PostProbMap_nparray_jit(self, CostMapProb,Mu,Sig,Phi_l,LookupTable_ProbCt,map_length,map_width,L,K): #,IndexMap):
        PostProbMap = np.array([ [ self.PostProb_ij([width, length],Mu,Sig,Phi_l,LookupTable_ProbCt,map_length,map_width,L,K, CostMapProb) for width in xrange(map_width) ] for length in xrange(map_length) ])
        return CostMapProb * PostProbMap



    