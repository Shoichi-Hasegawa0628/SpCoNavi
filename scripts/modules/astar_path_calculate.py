#!/usr/bin/env python
# coding:utf-8　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　#coding:utf-8

# Standard Library
import collections

# Third Party
import numpy as np
from scipy.stats import chi2, multivariate_normal, multinomial
import matplotlib.pyplot as plt

# Self-made Modules
from __init__ import *
import dataset
import converter

dataset_func = dataset.DataSet()
convert_func = converter.Converter()

class AstarPathCalculate():

    # The moving distance of the pathを計算する
    def PathDistance(self, Path):
        Distance = len(collections.Counter(Path))
        print("Path Distance is ", Distance)
        return Distance


    def Sampling_goal(self, Otb_B, THETA):
        # THETAを展開
        W, W_index, Myu, S, pi, phi_l, K, L = THETA

        # Prob math func of p(it | Otb_B, THETA) = Σc p(it | phi_c)p(st=Otb_B | Wc)p(c | pi)
        pmf_it = np.ones(K)
        for i in range(K):
            sum_Ct = np.sum([phi_l[c][i] * multinomial.pmf(Otb_B, sum(Otb_B), W[c]) * pi[c] for c in range(L)])
            pmf_it[i] = sum_Ct

        # Normalization
        pmf_it_n = np.array([pmf_it[i] / float(np.sum(pmf_it)) for i in range(K)])

        # Sampling it from multinomial distribution
        sample_it = multinomial.rvs(Sampling_J, pmf_it_n, size=1, random_state=None)
        print(sample_it)
        goal_candidate = []
        for it in range(K):
            count_it = 0
            while (count_it < sample_it[0][it]):
                goal_candidate += [convert_func.Map_coordinates_To_Array_index(
                    multivariate_normal.rvs(mean=Myu[it], cov=S[it], size=1, random_state=None))]
                count_it += 1
        # Xt_max = Map_coordinates_To_Array_index( [ Xp[pox.index(max(pox))][0], Xp[pox.index(max(pox))][1] ] ) #[0.0,0.0] ##確率最大の座標候補
        goal_candidate_tuple = [(goal_candidate[j][1], goal_candidate[j][0]) for j in range(Sampling_J)]
        print("Goal candidates:", goal_candidate_tuple)
        return goal_candidate_tuple


    def right(self, pos):
        return (pos[0], pos[1] + 1)


    def left(self, pos):
        return (pos[0], pos[1] - 1)


    def up(self, pos):
        return (pos[0] - 1, pos[1])


    def down(self, pos):
        return (pos[0] + 1, pos[1])


    def stay(self, pos):
        return (pos[0], pos[1])


    def Manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


    def astar_path_planner(self, THETA, W_index, outputfile, speech_num, start):
        maze = dataset_func.ReadMap(outputfile)
        height, width = maze.shape

        action_functions = [self.right, self.left, self.up, self.down,
                            self.stay]  # , migiue, hidariue, migisita, hidarisita]
        cost_of_actions = np.log(np.ones(len(action_functions)) / float(
            len(action_functions)))  # [    1/5,    1/5,  1/5,    1/5,    1/5]) #, ,    1,        1,        1,          1]

        #####Estimate the goal point by spatial concept
        Otb_B = [int(W_index[i] == Goal_Word[int(speech_num)]) * N_best for i in range(len(W_index))]
        print("BoW:", Otb_B)

        # Path-Planning
        # Path, Path_ROS, PathWeightMap, Path_one = PathPlanner(Otb_B, Start_Position[int(init_position_num)], THETA, CostMapProb) #gridmap, costmap)

        # Read the emission probability file
        PathWeightMap = dataset_func.ReadProbMap(outputfile, speech_num)

        #####描画
        # plt.imshow(maze, cmap="binary")
        gridmap = maze
        plt.imshow(gridmap + (40 + 1) * (gridmap == -1), origin='lower', cmap='binary', vmin=0, vmax=100,
                   interpolation='none')  # , vmin = 0.0, vmax = 1.0)

        plt.xticks(rotation=90)
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.tick_params(axis='y', which='major', labelsize=8)
        # plt.xlim([380,800])             #x軸の範囲
        # plt.ylim([180,510])             #y軸の範囲
        plt.xlabel('X', fontsize=10)
        plt.ylabel('Y', fontsize=10)
        # plt.xticks(np.arange(width), np.arange(width))
        # plt.yticks(np.arange(height), np.arange(height))
        plt.gca().set_aspect('equal')

        ###goalの候補を複数個用意する
        goal_candidate = self.Sampling_goal(Otb_B, THETA)  # (0,0)
        J = len(goal_candidate)
        if (J != THETA[6]):
            print("[WARNING] J is not K", J, K)
        p_cost_candidate = [0.0 for j in range(J)]
        Path_candidate = [[0.0] for j in range(J)]

        ###goal候補ごとにA*を実行
        for gc_index in range(J):
            goal = goal_candidate[gc_index]
            if (maze[goal[0]][goal[1]] != 0):
                print("[ERROR] goal", maze[goal[0]][goal[1]], "is not 0.")

            ###START A*
            open_list = []
            open_list_cost = []
            open_list_key = []
            closed_list = []
            closed_list_cost = []
            closed_list_key = []
            open_list.append(start)
            open_list_cost.append(0)
            open_list_key.append(0 + self.Manhattan_distance(start, goal))
            OYA = {}
            ko = (0), (0)
            Path = []

            while open_list:
                sorted_idx = np.argsort(open_list_key, kind="stable")
                pop_idx = sorted_idx[0]
                p = open_list.pop(pop_idx)
                p_cost = open_list_cost.pop(pop_idx)
                p_key = open_list_key.pop(pop_idx)
                closed_list.append(p)
                closed_list_cost.append(p_cost)
                closed_list_key.append(p_key)
                if p == goal:
                    break
                for act_func, act_cost in zip(action_functions, cost_of_actions):
                    q = act_func(p)
                    if (int(maze[q]) != 0):
                        continue
                    q_cost = p_cost - act_cost - np.log(PathWeightMap[q[0]][q[1]])  # current sum cost and action cost
                    q_pev = self.Manhattan_distance(q, goal) * np.log(
                        float(len(action_functions)))  # heuristic function
                    q_key = q_cost - q_pev

                    if q in open_list:
                        idx = open_list.index(q)
                        key = open_list_key[idx]
                        if key > q_key:
                            open_list_key[idx] = q_key
                            open_list_cost[idx] = q_cost
                    elif q in closed_list:
                        idx = closed_list.index(q)
                        key = closed_list_key[idx]
                        if key > q_key:
                            closed_list.pop(idx)
                            closed_list_cost.pop(idx)
                            closed_list_key.pop(idx)
                            open_list.append(q)
                            open_list_cost.append(q_cost)
                            open_list_key.append(q_key)
                            # plt.quiver(p[1], p[0], (q[1]-p[1]), (q[0]-p[0]), angles='xy', scale_units='xy', scale=1, color="tab:red")
                            OYA[(q[1], q[0])] = (p[1], p[0])
                            ko = (q[1]), (q[0])
                            # print(ko)
                    else:
                        open_list.append(q)
                        open_list_cost.append(q_cost)
                        open_list_key.append(q_key)
                        # plt.quiver(p[1], p[0], (q[1]-p[1]), (q[0]-p[0]), angles='xy', scale_units='xy', scale=1, color="tab:red")
                        OYA[(q[1], q[0])] = (p[1], p[0])
                        ko = (q[1]), (q[0])
                        # print(ko)

            # 最適経路の決定: ゴールから親ノード（どこから来たか）を順次たどっていく
            # i = len(OYA)
            # for oyako in reversed(OYA):
            ko_origin = ko
            ko = (goal[1], goal[0])
            print(ko, goal)
            # for i in range(p_cost):
            while (ko != (start[1], start[0])):
                # print(OYA[ko])
                try:
                    Path = Path + [OYA[ko]]
                except KeyError:
                    ko = ko_origin
                    Path = Path + [OYA[ko]]
                    print("NOT END GOAL.")

                ko = OYA[ko]
                # i = len(Path)
                # print(i, ko)
                # i -= 1

            print(goal, ": Total cost using A* algorithm is " + str(p_cost))
            p_cost_candidate[gc_index] = p_cost / float(len(Path))
            Path_candidate[gc_index] = Path

            ### select the goal of expected cost
        expect_gc_index = np.argmin(p_cost_candidate)
        Path = Path_candidate[expect_gc_index]
        goal = goal_candidate[expect_gc_index]
        print("Goal:", goal)
        return Path, goal, PathWeightMap
