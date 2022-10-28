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


# 实验数据控制
n=400
sub_count=4
device = torch.device('cpu')
network='random'
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

# 生成用于D-METIS的adjncy列表#####
sA = sp.csr_matrix(A)
#print(sA)
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
    solution_numerical = ode.odeint(SisDynamics(A), x0, t, method='dopri5')
    print("node dynamics:",solution_numerical.shape)

# 动态过程变量压缩，生成单独的节点权重vweight
true_y1 = solution_numerical.squeeze().t()
true_y1 = true_y1.numpy()
x_d=[]
for i in range(0,len(true_y1)):
    a=0
    for j in range(0,len(true_y1[i,:])):
        if j < (time_tick * 1.2 -1):
            a += abs(true_y1[i,j+1]-true_y1[i,j])*100  # 扩大一百倍，是为了取整后仍保持较好的约束精度，可以调整
    x_d.append(a)
x_d1 = list(map(int, x_d[:]))
############################################

(edgecuts,parts)=pymetis.part_graph(nparts=sub_count,adjncy=adjncy,xadj=xadj,vweights=x_d1)
A1=A

print('total_edges:',A1.sum()/2)
print("D-METIS results--",
      "edgecuts:",edgecuts,"belong_to:",parts)



# 切分成小图后  循环对各小图执行NDCN
for i in range(0,sub_count):
    node_sub=[]
    # print(i)
    for j in range(0,len(parts)):
        if i==parts[j]:
            # print(j)
            node_sub.append(j)
    print(node_sub)
    indices = torch.tensor(node_sub)
    a = torch.index_select(A1, 0, indices)
    b = torch.index_select(a, 1, indices)
    A=b
    print(A)

    #  GNNs 参数设置
    hidden,dropout,operator=20,0,'norm_lap'
    D = torch.diag(A.sum(1))
    L = (D - A)
    baseline='ndcn'


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
        print('Graph Operator[Default]: Normalized Laplacian')
        OM = torch.FloatTensor(normalized_laplacian(A.numpy()))  # L # normalized_adj

    if True:
        # For small network, dense matrix is faster
        # For large network, sparse matrix cause less memory
        L = torch_sensor_to_torch_sparse_tensor(L)
        A = torch_sensor_to_torch_sparse_tensor(A)
        OM = torch_sensor_to_torch_sparse_tensor(OM)

    # 恢复子图节点的动态过程值及其初始状态值
    true_y1 = solution_numerical.squeeze().t().to(device)  # 120 * 1 * 400  --squeeze--> 120 * 400 -t-> 400 * 120
    print(true_y1.size())
    true_y = torch.index_select(true_y1, 0, indices)
    true_y0 = x0.to(device)  # 400 * 1
    true_y0 = torch.index_select(true_y0, 0, indices)
    true_y_train = true_y[:, id_train].to(device)  # 400*80  for train
    true_y_test = true_y[:, id_test].to(device)  # 400*20  for extrapolation prediction
    if sampled_time == 'irregular':
        true_y_test2 = true_y[:, id_test2].to(device)  # 400*20  for interpolation prediction
    # L = L.to(device)  # 400 * 400
    # OM = OM.to(device)  # 400 * 400
    # A = A.to(device)

    # Build model
    input_size = true_y0.shape[1]   # y0: 400*1 ,  input_size:1 节点的特征维度，此研究中属性维度为1
    hidden_size = hidden  # hidden  # 20 default  # [400 * 1 ] * [1 * 20] = 400 * 20
    dropout = dropout  # 0 default, not stochastic ODE
    num_classes = 1  # 1 for regression
    # Params for discrete models
    input_n_graph= true_y0.shape[0]
    hidden_size_gnn = 5
    hidden_size_rnn = 10

    rtol = 0.01
    atol = 0.001
    method = 'dopri5'

    flag_model_type = ""  # "continuous" "discrete"  input, model, output format are little different
    # Continuous time network dynamic models
    if baseline == 'ndcn':
        print('Choose model:' + baseline)
        flag_model_type = "continuous"
        model = NDCN(input_size=input_size, hidden_size=hidden_size, A=OM, num_classes=num_classes,
                     dropout=dropout, no_embed=False, no_graph=False, no_control=False,
                     rtol=rtol, atol=atol, method=method)
    elif baseline == 'no_embed':
        print('Choose model:' + baseline)
        flag_model_type = "continuous"
        model = NDCN(input_size=input_size, hidden_size=input_size, A=OM, num_classes=num_classes,
                     dropout=dropout, no_embed=True, no_graph=False, no_control=False,
                     rtol=rtol, atol=atol, method=method)
    elif baseline == 'no_control':
        print('Choose model:' + baseline)
        flag_model_type = "continuous"
        model = NDCN(input_size=input_size, hidden_size=hidden_size, A=OM, num_classes=num_classes,
                     dropout=dropout, no_embed=False, no_graph=False, no_control=True,
                     rtol=rtol, atol=atol, method=method)
    elif baseline == 'no_graph':
        print('Choose model:' + baseline)
        flag_model_type = "continuous"
        model = NDCN(input_size=input_size, hidden_size=hidden_size, A=OM, num_classes=num_classes,
                     dropout=dropout, no_embed=False, no_graph=True, no_control=False,
                     rtol=rtol, atol=atol, method=method)
    # Discrete time or Sequential network dynamic models
    elif baseline == 'lstm_gnn':
        print('Choose model:' + baseline)
        flag_model_type = "discrete"
        # print('Graph Operator: Kipf') # Using GCN as graph embedding layer
        # OM = torch.FloatTensor(zipf_smoothing(A.numpy()))
        # OM = OM.to(device)
        model = TemporalGCN(input_size, hidden_size_gnn, input_n_graph, hidden_size_rnn, OM, dropout=dropout, rnn_type='lstm')
    elif baseline == 'gru_gnn':
        print('Choose model:' + baseline)
        flag_model_type = "discrete"
        model = TemporalGCN(input_size, hidden_size_gnn, input_n_graph, hidden_size_rnn, OM, dropout=dropout, rnn_type='gru')
    elif baseline == 'rnn_gnn':
        print('Choose model:' + baseline)
        flag_model_type = "discrete"
        model = TemporalGCN(input_size, hidden_size_gnn, input_n_graph, hidden_size_rnn, OM, dropout=dropout, rnn_type='rnn')


    model = model.to(device)
    # model = nn.Sequential(*embedding_layer, *neural_dynamic_layer, *semantic_layer).to(device)

    num_paras = get_parameter_number(model)
    viz,dump=1,1
    lr=0.01
    weight_decay=0.01
    niters=1000
    test_freq=20

    if viz:
        dirname = r'figure/mutualistic/' + network
        makedirs(dirname)
        fig_title = r'Mutualistic Dynamics'

    if dump:
        results_dir = r'results/mutualistic/' + network
        makedirs(results_dir)

    if __name__ == '__main__':
        t_start = time.time()
        params = model.parameters()
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        criterion = F.l1_loss  # F.mse_loss(pred_y, true_y)
        # time_meter = RunningAverageMeter(0.97)
        # loss_meter = RunningAverageMeter(0.97)
        if dump:
            results_dict = {
                'v_iter': [],
                'abs_error': [],
                'rel_error': [],
                'true_y': [solution_numerical.squeeze().t()],
                'predict_y': [],
                'abs_error2': [],
                'rel_error2': [],
                'predict_y2': [],
                'model_state_dict': [],
                'total_time': []}

        for itr in range(1, niters + 1):
            optimizer.zero_grad()
            # 用于定义losstrain的
            if flag_model_type == "continuous":
                pred_y = model(t_train, true_y0)  # 80 * 400 * 1 should be 400 * 80
                pred_y = pred_y.squeeze().t()
                loss_train = criterion(pred_y, true_y_train) # true_y)  # 400 * 20 (time_tick)
                # torch.mean(torch.abs(pred_y - batch_y))
                relative_loss_train = criterion(pred_y, true_y_train) / true_y_train.mean()
            elif flag_model_type == "discrete":
                # true_y_train = true_y[:, id_train]  # 400*80  for train
                pred_y = model(true_y_train[:, :-1])  # true_y_train 400*80 true_y_train[:, :-1] 400*79
                # pred_y = pred_y.squeeze().t()
                loss_train = criterion(pred_y, true_y_train[:, 1:])  # true_y)  # 400 * 20 (time_tick)
                # torch.mean(torch.abs(pred_y - batch_y))
                relative_loss_train = criterion(pred_y, true_y_train[:, 1:]) / true_y_train[:, 1:].mean()
            else:
                print("flag_model_type NOT DEFINED!")
                exit(-1)

            loss_train.backward()
            optimizer.step()

            # time_meter.update(time.time() - t_start)
            # loss_meter.update(loss.item())

            # 这里开始才是开始迭代循环训练和预测的功能

            if itr % test_freq == 0:
                with torch.no_grad():
                    if flag_model_type == "continuous":
                        # pred_y = model(true_y0).squeeze().t() # odeint(model, true_y0, t)
                        # loss = criterion(pred_y, true_y)
                        # relative_loss = criterion(pred_y, true_y) / true_y.mean()
                        pred_y = model(t, true_y0).squeeze().t()  # odeint(model, true_y0, t)
                        print((pred_y.size()))

                        loss = criterion(pred_y[:, id_test], true_y_test)
                        relative_loss = criterion(pred_y[:, id_test], true_y_test) / true_y_test.mean()
                        if sampled_time == 'irregular': # for interpolation results
                            loss2 = criterion(pred_y[:, id_test2], true_y_test2)
                            relative_loss2 = criterion(pred_y[:, id_test2], true_y_test2) / true_y_test2.mean()
                    elif flag_model_type == "discrete":
                        pred_y = model(true_y_train, future=len(id_test)) #400*100
                        # pred_y = pred_y.squeeze().t()
                        loss = criterion(pred_y[:, id_test], true_y_test) #pred_y[:, id_test] 400*20
                        # torch.mean(torch.abs(pred_y - batch_y))
                        relative_loss = criterion(pred_y[:, id_test], true_y_test) / true_y_test.mean()

                    if dump:
                        # Info to dump
                        results_dict['v_iter'].append(itr)
                        results_dict['abs_error'].append(loss.item())    # {'abs_error': [], 'rel_error': [], 'X_t': []}
                        results_dict['rel_error'].append(relative_loss.item())
                        results_dict['predict_y'].append(pred_y[:, id_test])
                        results_dict['model_state_dict'].append(model.state_dict())
                        if sampled_time == 'irregular':  # for interpolation results
                            results_dict['abs_error2'].append(loss2.item())  # {'abs_error': [], 'rel_error': [], 'X_t': []}
                            results_dict['rel_error2'].append(relative_loss2.item())
                            results_dict['predict_y2'].append(pred_y[:, id_test2])
                        # now = datetime.datetime.now()
                        # appendix = now.strftime("%m%d-%H%M%S")
                        # results_dict_path = results_dir + r'/result_' + appendix + '.' + dump_appendix
                        # torch.save(results_dict, results_dict_path)
                        # print('Dump results as: ' + results_dict_path)
                    if sampled_time == 'irregular':
                        print('Iter {:04d}| Train Loss {:.6f}({:.6f} Relative) '
                              '| Test Loss {:.6f}({:.6f} Relative) '
                              '| Test Loss2 {:.6f}({:.6f} Relative) '
                              '| Time {:.4f}'
                              .format(itr, loss_train.item(), relative_loss_train.item(),
                                      loss.item(), relative_loss.item(),
                                      loss2.item(), relative_loss2.item(),
                                      time.time() - t_start))
                    else:
                        print('Iter {:04d}| Train Loss {:.6f}({:.6f} Relative) '
                              '| Test Loss {:.6f}({:.6f} Relative) '
                              '| Time {:.4f}'
                              .format(itr, loss_train.item(), relative_loss_train.item(),
                                      loss.item(), relative_loss.item(),
                                      time.time() - t_start))

        now = datetime.datetime.now()
        appendix = now.strftime("%m%d-%H%M%S")
        with torch.no_grad():
            if flag_model_type == "continuous":
                pred_y = model(t, true_y0).squeeze().t()  # odeint(model, true_y0, t)
                loss = criterion(pred_y[:, id_test], true_y_test)
                relative_loss = criterion(pred_y[:, id_test], true_y_test) / true_y_test.mean()
                if sampled_time == 'irregular':  # for interpolation results
                    loss2 = criterion(pred_y[:, id_test2], true_y_test2)
                    relative_loss2 = criterion(pred_y[:, id_test2], true_y_test2) / true_y_test2.mean()
            elif flag_model_type == "discrete":
                pred_y = model(true_y_train, future=len(id_test))  # 400*100
                loss = criterion(pred_y[:, id_test], true_y_test)  # pred_y[:, id_test] 400*20
                relative_loss = criterion(pred_y[:, id_test], true_y_test) / true_y_test.mean()

            if sampled_time == 'irregular':
                print('Iter {:04d}| Train Loss {:.6f}({:.6f} Relative) '
                      '| Test Loss {:.6f}({:.6f} Relative) '
                      '| Test Loss2 {:.6f}({:.6f} Relative) '
                      '| Time {:.4f}'
                      .format(itr, loss_train.item(), relative_loss_train.item(),
                              loss.item(), relative_loss.item(),
                              loss2.item(), relative_loss2.item(),
                              time.time() - t_start))
            else:
                print('Iter {:04d}| Train Loss {:.6f}({:.6f} Relative) '
                      '| Test Loss {:.6f}({:.6f} Relative) '
                      '| Time {:.4f}'
                      .format(itr, loss_train.item(), relative_loss_train.item(),
                              loss.item(), relative_loss.item(),
                              time.time() - t_start))

            t_total = time.time() - t_start
            print('Total Time {:.4f}'.format(t_total))
            num_paras = get_parameter_number(model)
            if dump:
                results_dict['total_time'] = t_total
                results_dict_path = results_dir + r'/result_' + appendix + '.' + baseline  #dump_appendix
                torch.save(results_dict, results_dict_path)
                print('Dump results as: ' + results_dict_path)

                # Test dumped results:
                rr = torch.load(results_dict_path)
                fig, ax = plt.subplots()
                ax.plot(rr['v_iter'], rr['abs_error'], '-', label='Absolute Error')
                ax.plot(rr['v_iter'], rr['rel_error'], '--', label='Relative Error')
                legend = ax.legend( fontsize='x-large') # loc='upper right', shadow=True,
                # legend.get_frame().set_facecolor('C0')
                fig.savefig(results_dict_path + ".png", transparent=True)
                fig.savefig(results_dict_path + ".pdf", transparent=True)

