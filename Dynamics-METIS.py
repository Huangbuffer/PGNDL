from GNNs import *
import networkx as nx
import time
import datetime
import pymetis
from dynamics_for_sim import *
from utils import *
import os
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device('cpu')

# D-Dynamics 实验数据控制
n=400
sub_count=4
# n=2000
# sub_count=8


seed=0
layout='community'
sampled_time = 'irregular'
time_tick=100
T=5
for network in ['random']: # ['random','power_law', 'small_world', 'community']
    for Dynamics, Da in zip([GeneDynamics], # [SisDynamics,MutualDynamics, GeneDynamics]
                            ['GeneDynamics']): # ['SisDynamics','MutualDynamics', 'GeneDynamics']

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
            n1 = int(n / 3)
            n2 = int(n / 3)
            n3 = int(n / 4)
            n4 = n - n1 - n2 - n3
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
        x0 = torch.randint(0,30,(n,))
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
            solution_numerical = ode.odeint(Dynamics(A), x0, t, method='dopri5')
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
                    a += abs(true_y1[i,j+1]-true_y1[i,j])*1000  # 扩大一百倍，是为了取整后仍保持较好的约束精度，可以调整
            x_d.append(a)
        x_d1 = list(map(int, x_d[:]))
        ############################################
        true_y1 = solution_numerical.squeeze().t().to(device)
        print("Raw dynamic graph‘ size："'\n',A.numpy())
        print('Dynamincs Compression done!')

        ###########################################
        edgecuts,parts=pymetis.part_graph(nparts=sub_count,adjncy=adjncy,xadj=xadj) # vweights=x_d1
        A1=A
        edgecount=A1.sum()/2
        print('Total_edges of Raw Graph:',edgecount,'\n')
        print("------------------D-METIS results------------------")
        print("DMETIS_edgecuts:",edgecuts,'\n',"DMETIS_subgraph_counts:",sub_count)
        print("Partitioning Results:",parts)
        print("------------------D-METIS results------------------")


        # 设置 OM 计算方式（用于传入GNNs模型）
        operator='norm_lap'

        # 切分成小图后  循环保存对各小图的OM、node_states文件，便于并行调用PGNDL模块
        for i in range(0,sub_count):
            node_sub=[]
            # print(i)
            for j in range(0,len(parts)):
                if i==parts[j]:
                    # print(j)
                    node_sub.append(j)
            # print(node_sub)
            ################## 恢复子图i的网络结构
            indices = torch.tensor(node_sub)
            a = torch.index_select(A1, 0, indices)
            b = torch.index_select(a, 1, indices)
            A=b
            # print(A)

            D = torch.diag(A.sum(1))
            L = (D - A)
            print('Size of subgraph'+str(i),A.size(),'Edges of subgraph'+str(i),A.sum()/2) # 查看子图i规模
            edgecount-=A.sum()/2
            print(int(edgecount))
            if operator == 'lap':
                print('Graph Operator: Laplacian')
                OM = L
            elif operator == 'kipf':
                print('Graph Operator: Kipf')
                OM = torch.FloatTensor(zipf_smoothing(A.numpy()))
            elif operator == 'norm_adj':
                print('Graph Operator: Normalized Adjacency')
                OM = torch.FloatTensor(normalized_adj(A.numpy()))
            else:
                print('Graph Operator: Normalized Laplacian')
                OM = torch.FloatTensor(normalized_laplacian(A.numpy()))  # L # normalized_adj

            if True:
                # For small network, dense matrix is faster
                # For large network, sparse matrix cause less memory
                L = torch_sensor_to_torch_sparse_tensor(L)
                A = torch_sensor_to_torch_sparse_tensor(A)
                OM = torch_sensor_to_torch_sparse_tensor(OM)
            ############################# 保存数据用于模型的训练和预测
            # 先转成dense tensor，在使用OM时要先转回sparse tensor
            OM_D = OM.to_dense()
            # print('OM_type:', type(OM), '\n', 'OM:', OM,'\n', 'OM_D:',OM_D)
            # 文件夹定义
            subgraph_dir = r'graph/' + str(network)+ str(Da) + str(n) + '/OM'
            makedirs(subgraph_dir)
            pth = subgraph_dir + '/' + str(i)
            # 保存OM_D的npy文件
            np.save(pth+'.npy',OM_D.numpy())

            ##################### 恢复子图i的动态过程
            true_y = torch.index_select(true_y1, 0, indices)
            true_y0 = x0  # 400 * 1
            true_y0 = torch.index_select(true_y0, 0, indices)
            true_y_train = true_y[:, id_train]  # 400*80  for train
            true_y_test = true_y[:, id_test]  # 400*20  for extrapolation prediction
            # print(true_y0,true_y)

            # 生成node_states文件
            subgraph_dir1 = r'graph/'+str(network)+str(Da) +str(n)+'/node_states'
            makedirs(subgraph_dir1)

            # 保存节点初始状态
            pth=subgraph_dir1+'/'+ str(i)+'true_y0'
            torch.save(true_y0, pth+'.pt')

            # 保存节点全部动态过程
            pth=subgraph_dir1+'/'+ str(i)+'true_y'
            torch.save(true_y, pth+'.pt')
            # 保存节点用于训练的动态过程
            pth=subgraph_dir1+'/'+ str(i)+'true_y_train'
            torch.save(true_y_train, pth+'.pt')
            # 保存节点用于测试的动态过程
            pth=subgraph_dir1+'/'+ str(i)+'true_y_test'
            torch.save(true_y_test, pth+'.pt')
            # 保存节点用于测试的动态过程（irregular）
            if sampled_time == 'irregular':
                true_y_test2 = true_y[:, id_test2].to(device)  # 400*20  for interpolation prediction
                pth=subgraph_dir1+'/'+ str(i)+'true_y_test2'
                torch.save(true_y_test2, pth+'.pt')