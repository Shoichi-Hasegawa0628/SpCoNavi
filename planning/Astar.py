#coding:utf-8

###########################################################
# Path-Planning Program by A star algorithm (開発中)
# Akira Taniguchi 2019/06/24-2019/06/27
# Spacial Thanks ; Ryo Ozaki
###########################################################

import sys
import numpy as np
import matplotlib.pyplot as plt

maze_file = "../CoRL/data/s1DK_01/navi/s1DK_01.pgm" #"./sankou/sample_maze.txt"

maze = np.zeros((192,192))
height, width = maze.shape

##########
#PGMファイルの読み込み
#http://www.not-enough.org/abe/manual/api-aa09/fileio2.html
infile = open(maze_file , 'rb') #sys.argv[1]
#outfile = open(sys.argv[2], 'w')

for i in range(4): #最初の4行は無視
    d = infile.readline()
    print(d)
    #outfile.write(d)

#data = infile.read()#.decode('utf-8',"ignore")
#print(data)
#maze = np.array(data) #, dtype="int")  #文字列として読み込まれることに注意

for h in range(height):
    for w in range(width):
        d = infile.read(1)
        maze[h][w] = int(255 - ord(d))/255
"""
while True:
    d = infile.read(1)
    if len(d) == 0:
        break\
    #print( ord(d) )
    #print('(%x) ' % ord(d))
"""
infile.close
##########


#maze = np.loadtxt(maze_file, dtype=int)
height, width = maze.shape

start = (83,39) #(92,126) #(126,92) #(1, 1)
goal = (95,41) #(97,55) #(55,97) #(height-2, width-2)

def right(pos):
    return (pos[0], pos[1] + 1)

def left(pos):
    return (pos[0], pos[1] - 1)

def up(pos):
    return (pos[0] - 1, pos[1])

def down(pos):
    return (pos[0] + 1, pos[1])

action_functions = [right, left, up, down]
cost_of_actions  = [    1,    1,  1,    1]

def Manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

plt.imshow(maze, cmap="binary")
plt.xticks(rotation=90)
#plt.xticks(np.arange(width), np.arange(width))
#plt.yticks(np.arange(height), np.arange(height))
plt.gca().set_aspect('equal')

# スタートとゴールをプロットする
plt.plot(start[1], start[0], "D", color="tab:blue", markersize=1)
plt.plot(goal[1], goal[0], "D", color="tab:green", markersize=1)

open_list = []
open_list_cost = []
open_list_key = []
closed_list = []
closed_list_cost = []
closed_list_key = []
open_list.append(start)
open_list_cost.append(0)
open_list_key.append(0 + Manhattan_distance(start, goal))
OYA = {}
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
        if maze[q] == 1:
            continue
        q_cost = p_cost + act_cost
        q_pev = Manhattan_distance(q, goal)
        q_key = q_cost + q_pev

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
                #plt.quiver(p[1], p[0], (q[1]-p[1]), (q[0]-p[0]), angles='xy', scale_units='xy', scale=1, color="tab:red")
                OYA[(q[1], q[0])] = (p[1], p[0])
                ko = (q[1]), (q[0])
                #print(ko)
        else:
            open_list.append(q)
            open_list_cost.append(q_cost)
            open_list_key.append(q_key)
            #plt.quiver(p[1], p[0], (q[1]-p[1]), (q[0]-p[0]), angles='xy', scale_units='xy', scale=1, color="tab:red")
            OYA[(q[1], q[0])] = (p[1], p[0])
            ko = (q[1]), (q[0])
            #print(ko)

#最適経路の決定：ゴールから親ノード（どこから来たか）を順次たどっていく
#i = len(OYA)
#for oyako in reversed(OYA):
print(ko,goal)
for i in range(p_cost-1):
  #print(OYA[ko])
  Path = Path + [OYA[ko]]
  ko = OYA[ko]
  #i -= 1
for i in range(len(Path)):
  plt.plot(Path[i][0], Path[i][1], "D", color="tab:red", markersize=1)

print("Total cost using A* algorithm is "+ str(p_cost))
plt.show()
