import matplotlib
import numpy as np
from  matplotlib import pyplot as plt
import pymetis
from dynamics_for_sim import *
from utils import *
import os
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# D-Dynamics 实验数据控制
n=400
sub_range=range(4,13)
device = torch.device('cpu')
network='power_law'
seed=0
layout='community'
sampled_time = 'irregular'
time_tick=100
T=5

if network == 'random':
    print("Choose graph: " + network)
    G = nx.erdos_renyi_graph(n, 0.1, seed=seed)
    G = networkx_reorder_nodes(G, layout)
    A = torch.FloatTensor(nx.to_numpy_array(G))
elif network == 'power_law':
    print("Choose graph: " + network)
    G = nx.barabasi_albert_graph(n, 5, seed=seed)
    G = networkx_reorder_nodes(G, layout)
    A = torch.FloatTensor(nx.to_numpy_array(G))
elif network == 'small_world':
    print("Choose graph: " + network)
    G = nx.newman_watts_strogatz_graph(n, 5, 0.5, seed=seed)
    G = networkx_reorder_nodes(G, layout)
    A = torch.FloatTensor(nx.to_numpy_array(G))
elif network == 'community':
    print("Choose graph: " + network)
    n1 = int(n/3)
    n2 = int(n/3)
    n3 = int(n/4)
    n4 = n - n1 - n2 -n3
    G = nx.random_partition_graph([n1, n2, n3, n4], .25, .01, seed=seed)
    G = networkx_reorder_nodes(G, layout)
    A = torch.FloatTensor(nx.to_numpy_array(G))
print('-------------Dynaminc-Graph simulation---------')

# 生成用于D-METIS的adjncy列表#####
sA = sp.csr_matrix(A)
so=sA.tocoo()
#print(so.row)
#print(so.col)
adjncy=so.col.tolist()
########################################
# xadj=[0]
# a=1
# for i in range(0,len(so.row.tolist())):
#     if i < len(so.row)-1:
#         if so.row[i+1] == so.row[i]:
#             a=1+a
#         else:
#             xadj.append(a)
# xadj # useless
print("Start the D-METIS...")
# 从邻接矩阵-->D-metis所需的切分序列xadj
xadj=[0]
a=0
for x in A:
    a=sum(x)
    xadj.append(a)
r=np.array(xadj)
xadj=r.cumsum()
xadj=list(map(int, xadj))
print(xadj)
###############################################
#  生成初始值x0
x0 = torch.randint(0,8,(n,))
x0 = x0.view(-1, 1).float()
print('Genertate the t=0 state')
# 生成时间切片（对时间节点采样）
# equally-sampled time

if sampled_time == 'equal':
    print('Build Equally-sampled -time dynamics')
    t = torch.linspace(0., T, time_tick)  # time_tick) # 100 vector
    # train_deli = 80
    id_train = list(range(int(time_tick * 0.8))) # first 80 % for train
    id_test = list(range(int(time_tick * 0.8), time_tick)) # last 20 % for test (extrapolation)
    t_train = t[id_train]
    t_test = t[id_test]
elif sampled_time == 'irregular':
    print('Build irregularly-sampled -time dynamics')
    # irregular time sequence
    sparse_scale = 10
    t = torch.linspace(0., T, time_tick * sparse_scale) # 100 * 10 = 1000 equally-sampled tick
    t = np.random.permutation(t)[:int(time_tick * 1.2)]
    t = torch.tensor(np.sort(t))
    t[0] = 0
    # t is a 120 dim irregularly-sampled time stamps

    id_test = list(range(time_tick, int(time_tick * 1.2)))  # last 20 beyond 100 for test (extrapolation)
    id_test2 = np.random.permutation(range(1, time_tick))[:int(time_tick * 0.2)].tolist()
    id_test2.sort() # first 20  in 100 for interpolation
    id_train = list(set(range(time_tick)) - set(id_test2))  # first 80 in 100 for train
    id_train.sort()

    t_train = t[id_train]
    t_test = t[id_test]
    t_test2 = t[id_test2]

# 生成0-T的网络节点状态 ###########
with torch.no_grad():
    solution_numerical = ode.odeint(MutualDynamics(A), x0, t, method='dopri5')
    ############### 选择动力学函数
    print("node dynamics:",solution_numerical.shape)
print('--------------Raw graph‘ dynamic simulation done!------------------')

# 动态过程变量压缩，生成单独的节点权重vweight
true_y1 = solution_numerical.squeeze().t().to(device)  # 120 * 1 * 400  --squeeze--> 120 * 400 -t-> 400 * 120
true_y1 = true_y1.numpy()
x_d=[]
for i in range(0,len(true_y1)):
    a=0
    for j in range(0,len(true_y1[i,:])):
        if j < (time_tick * 1.2 -1):
            a += abs(true_y1[i,j+1]-true_y1[i,j])*10000  # 扩大一百倍，是为了取整后仍保持较好的约束精度，可以调整
    x_d.append(a)
x_d1 = list(map(int, x_d[:]))

############################################
true_y1 = solution_numerical.squeeze().t().to(device)
print("Raw dynamic graph‘ size："'\n',A.numpy())
print('Dynamincs Compression done!')

###########################################
A1 = A
edgecount = A1.sum() / 2
print('Total_edges of Raw Graph:', edgecount, '\n')
print("------------------D-METIS results------------------")

# plot setup
anal='Vertex Distribution'
# anal='Dynamic Change'
plt.figure(figsize=(9, 9))

edge_all_subgraphs=[]
edgecut_list=[]
edgecut_list1=[]
for sub_count in sub_range:
    edgecuts,parts=pymetis.part_graph(nparts=sub_count,adjncy=adjncy,xadj=xadj,vweights=x_d1)
    edgecuts1, parts1 = pymetis.part_graph(nparts=sub_count, adjncy=adjncy, xadj=xadj)
    parts2 = np.random.randint(sub_count,size=n) #随机切割图的结果
    edgecut_list.append(edgecuts)
    edgecut_list1.append(edgecuts1)
    print("DMETIS_edgecuts:", edgecuts, '\n', "DMETIS_subgraph_counts:", sub_count)
    # print("Partitioning Results:", parts)
    print("------------------D-METIS results------------------")
    # 通用设置
    # matplotlib.rc('axes', facecolor='white')
    # matplotlib.rc('figure', figsize=(6, 4))
    # matplotlib.rc('axes', grid=False)
    # 数据及线属性

    # 切分成小图后  循环保存对各小图的OM、node_states文件，便于并行调用PGNDL模块
    sub_weight = [] # 各子图的权重和（动态变换累和）
    sub_nodes = [] # 各子图的节点数
    sub_weight1 = [] # 各子图的权重和（动态变换累和）
    sub_nodes1 = [] # 各子图的节点数
    sub_nodes2 = []
    sub_weight2 = []
    for i in range(0, sub_count):
        wk=0
        wk1=0
        wk2=0
        node1=0
        node=0
        node2=0
        # print(i)
        for j,k,q,r in zip(range(0, len(parts)),x_d1,range(0,len(parts1)),range(0,len(parts2))):
            if i == parts[j]:
                # print(j)
                wk+=k
                node+=1
            if i == parts1[q]:
                wk1+=k
                node1+=1
            if i == parts2[r]:
                wk2 += k
                node2 += 1
        sub_weight.append(wk/10000)
        sub_nodes.append(node)
        sub_weight1.append(wk1/10000)
        sub_nodes1.append(node1)
        sub_weight2.append(wk2/10000)
        sub_nodes2.append(node2)
    # print(sub_nodes)

    x = range(0, sub_count)
    y = sub_weight
    z = sub_nodes
    y1=sub_weight1
    z1=sub_nodes1
    y2=sub_weight2
    z2=sub_nodes2

    plt.subplot(3,3,sub_count-3)
    if anal == 'Vertex Distribution':

        plt.plot(x, z, 'o:r',markersize=4) # Dmetis

        plt.plot(x, z1, 'o:g',markersize=4) # Metis

        plt.plot(x, z2, 'o:y',markersize=4) # random
        my_y_ticks = np.arange(20, 120, 10)
        plt.yticks(my_y_ticks)
    else:
        plt.plot(x, y, '*:r')

        plt.plot(x, y1, '*:g')

        plt.plot(x, y2, '*:y',markersize=5)
        my_y_ticks = np.arange(100, 800, 100)
        plt.yticks(my_y_ticks)
    # 标题设置


plt.suptitle('Balance Analysis: '+str(anal),fontsize=12,x=0.5,y=0.93)

plt.legend(labels=['D-METIS','METIS','Random'], loc='upper center')
# plt.xlabel('Number of Subgraph')
# plt.ylabel('Number of Nodes/Dynamics-Changes in Subgraph')
plt.savefig('Balance Analysis of D-METIS'+str(anal)+'.svg')
plt.show()


print(edgecut_list)