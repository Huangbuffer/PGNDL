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
# from tet2 import *
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

n=400
sub_count=4
# n=2000
# sub_count=8

#  GNNs 参数设置
hidden,dropout,operator=20,0,'norm_lap'

baseline='ndcn'
seed=0
layout='community'
sampled_time = 'irregular'
time_tick=100
T=5
device='cpu'
# 文件夹定义
# ['random','power_law', 'small_world', 'community']
# [SisDynamics,MutualDynamics, GeneDynamics]
time_use={'random':[],'power_law':[], 'community':[],'small_world':[]}
for network in ['random','power_law', 'small_world', 'community']:
    for Da in ['SisDynamics','MutualDynamics', 'GeneDynamics']:
        print('network：',network)
        subgraph_dir = r'graph/' + str(network) +str(Da) + str(n) + '/OM'
        makedirs(subgraph_dir)
        pth = subgraph_dir + '/'
        subgraph_dir1 = r'graph/'+str(network)+str(Da)+str(n)+'/node_states/'
        los_sub=[]
        los_sub2=[]
        for i in range(0,sub_count):
            # 导入数据
            OM=torch.tensor(np.load(pth+str(i)+'.npy')).to_sparse()
            # print('recovered OM:', torch.tensor(np.load(pth + '.npy')).to_sparse())

            # 恢复子图节点的动态过程值及其初始状态值

            true_y = torch.load(subgraph_dir1+str(i)+'true_y.pt') # .to(device)
            true_y0 = torch.load(subgraph_dir1+str(i)+'true_y0.pt')

            true_y_train = torch.load(subgraph_dir1+str(i)+'true_y_train.pt')
            true_y_test = torch.load(subgraph_dir1+str(i)+'true_y_test.pt')
            if sampled_time == 'irregular':
                true_y_test2 = torch.load(subgraph_dir1+str(i)+'true_y_test2.pt')  # 400*20  for interpolation prediction

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
            method = 'euler'

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

            # if dump:
            #     results_dir = r'results/mutualistic/' + network
            #     makedirs(results_dir)

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
                    #print('Total Time {:.4f}'.format(t_total))
                    num_paras = get_parameter_number(model)
                    los_sub.append(min(results_dict['abs_error']))
                    los_sub2.append(min(results_dict['abs_error2']))
                    time_use[network].append(t_total)
        print("每个子图的训练时间：",time_use[network])
        print(str(Da)+str(network)+'的平均训练时间:',np.mean(time_use[network]))
        print('每个子图的最小预测平均绝对误差：',los_sub)
        print('每个子图的最小预测平均绝对误差2：', los_sub2)
        print(str(Da)+str(network) + '各子图最小误差的平均值(extrapolation):', np.mean(los_sub))
        print(str(Da)+str(network) + '各子图最小误差的平均值(interpolation):',np.mean(los_sub2))
    print(time_use)
