import numpy as np
import scipy.io as sio
import random
import torch
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import networkx as nx
import torch.optim as optim
from torch_geometric.data import Data,data, DataLoader
from torch_geometric.nn import GCNConv
from torch import conv2d
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import defaultdict
data = sio.loadmat(r"C:\Users\86156\Downloads\2024年湖南大学数学建模竞赛赛题及参赛须知\2024年湖南大学数学建模竞赛赛题及参赛须知\B题：药物属性预测\附件：MMM_data.mat")
adjacency_matrices = data['MMM']['am'][0]  # 邻接矩阵
node_labels = data['MMM']['al'][0]  # 节点标签
graph_labels=data['lable']
batch_size=40
# 设置随机数种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(15)  # 设置随机数种子为42

#将al进行处理，去除内部O格式转换成统一数组
def process_label(y):
    graph_len=y.shape[0]
    new_label = []
    # 处理y的最后一个维度，并将处理后的结果赋值给新数组的相应位置
    for i in range(graph_len):#遍历每张图
        node_len=y[i].shape[0]
        temp_node=[]
        for j in range(node_len):#每个节点
            temp_label=[0 for _ in range(node_len)]
            label = y[i][j][0] # 
            temp_node.append(label[0])  # 赋值给新数组的相应位置
        new_label.append(temp_node)
    return np.array(new_label)
node_labels=process_label(node_labels)

#单个图中将所有节点的特征向量维度对齐
def my_unsqueeze(data):
    # 将每个不定长度的数组填充成相同的长度
    max_length = max(len(arr) for arr in data)
    padded_j = [np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=0) for arr in data]
    # 创建一个零张量
    x = torch.zeros((len(data), max_length))
    # 将填充后的数组复制到零张量中
    for i, arr in enumerate(padded_j):
        x[i, :] = torch.tensor(arr, dtype=torch.float32)
    return x
# 构建图数据
graphs = []
x=[]
y=[]
edge_index=[]
for i in range(adjacency_matrices.shape[0]):
    #先将邻接矩阵统一到28个节点，再求取边索引，后续更改需要结合Data数据要求进行更改
    # ，后面第三行是为了求取不重复的边索引，由于边索引的对称性可知数据应该是无向图
    t=np.pad(adjacency_matrices[i],(0, 28-adjacency_matrices[i].shape[0]),mode='constant', constant_values=0)
    unique_edge_index = list(set(map(tuple, t.nonzero())))
    nonzero_indices=torch.tensor(unique_edge_index, dtype=torch.long)
    edge_index.append(nonzero_indices)

    x.append(my_unsqueeze(node_labels[i]))
    y.append(torch.tensor(graph_labels[i], dtype=torch.float32))

#对于所有图进行每个维度的统一，当输入为al时可求出邻接矩阵
def feature_process(data):
    max_dim2 = max(arr.shape[0] for arr in data)
    max_dim3 = max(arr.shape[1] for arr in data)
    padded_j = [np.pad(arr, ((0, max_dim2 - arr.shape[0]), (0,max_dim2-arr.shape[1])), 'constant', constant_values=0) for arr in data]
    # 创建一个零张量
    x = torch.zeros((len(data), max_dim2, max_dim2))
    # 将填充后的数组复制到零张量中x
    for i, arr in enumerate(padded_j):#单张图
        if max_dim3==max_dim2:
            x[i, :, :] = torch.tensor(arr, dtype=torch.float32)
        #由al求邻接矩阵
        else:
            for j, _ in enumerate(arr):#单个节点
                for k in _:
                    x[i,j,int(k)-1]=1
            x=torch.tensor(x, dtype=torch.float32)

    return x
# y=feature_process(y)
x=feature_process(x)
def count_rings_per_node(G):
    # 使用cycle_basis获取环的生成器
    rings = nx.cycle_basis(G)
    
    # 创建一个字典来存储每个节点经过的环的数目
    rings_per_node = defaultdict(int)
    
    # 遍历所有检测到的环
    for ring in rings:
        # 增加环中每个节点的环数目
        for node in ring:
            rings_per_node[node] += 1
    
    return dict(rings_per_node)
# 暂时没用到，可用于求取特征向量
def extract_features(edges):
    edges = list(zip(edges[1], edges[0]))
    features = []
    # 假设你有一个无向图
    G = nx.Graph()

    # 添加边到图中
    G.add_edges_from([(e[0].item(), e[1].item()) for e in edges])
    # nx.draw(G, with_labels=True, font_weight='bold')
    # plt.show()
    # 计算每个节点的度
    degrees = dict(G.degree())
    # print("Degrees:", degrees)
    # 计算每个节点的聚类系数
    clustering = dict(nx.clustering(G))
    rings_per_node = count_rings_per_node(G)
    # print("Clustering Coefficients:", clustering)
    # 创建一个特征矩阵，其中包含度和聚类系数作为特征
    for node in G.nodes():
        degree = degrees[node]
        clustering_coefficient = rings_per_node[node] if node in rings_per_node.keys() else 0
        features.append([degree, clustering_coefficient])
    return torch.tensor(features, dtype=torch.float32)
fea=[]
for i in range(188):
    fea.append(extract_features(edge_index[i]))

#Data要求数据集中edge_ndex必须为2xN，后续主要解决无法进行随机选取数据的问题
for i in range(adjacency_matrices.shape[0]):
    graphs.append(Data(x=fea[i], edge_index=edge_index[i], y=torch.tensor(y[i])))

# 目前使用的手写数据加载用到，后面使用随机数进行索引
train_loader=graphs
# train_loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(graphs[120:], batch_size=batch_size, shuffle=False)
# train_loader = graphs[:150]
# test_loader = graphs[150:]

##########################
# 这部分先不删，后续有时间可以写自定义数据加载
# class CustomDataset(Dataset):
#     def __init__(self, adjacency_matrices, node_labels, graph_labels):
#         # 初始化数据
#         self.adjacency_matrices = adjacency_matrices
#         self.node_labels = node_labels
#         self.graph_labels = graph_labels
#         self.edge_index = []
#         self.x = []
#         self.y = []

#         # 处理数据，填充特征，创建稀疏矩阵等
#         for i in range(adjacency_matrices.shape[0]):
#             nonzero_indices = torch.tensor(np.array(adjacency_matrices[i].nonzero()), dtype=torch.long).t()
#             nonzero_values = torch.tensor(adjacency_matrices[i][nonzero_indices[:, 0], nonzero_indices[:, 1]], dtype=torch.float32)
#             sparse_matrix = torch.sparse.FloatTensor(nonzero_indices.t(), nonzero_values, (28, 28))
#             self.edge_index.append(sparse_matrix)
#             self.x.append(my_unsqueeze(node_labels[i]))
#             self.y.append(torch.tensor(graph_labels[i], dtype=torch.float32))

#         # 填充特征矩阵
#         self.x = feature_process(self.x)

#     def __len__(self):
#         # 返回数据集中图的数量
#         return len(self.adjacency_matrices)

#     def __getitem__(self, idx):
#         # 根据索引返回一个图数据对象
#         return Data(x=self.x[idx], edge_index=self.edge_index[idx], y=self.y[idx])

# # 假设你已经有了初始化数据 adjacency_matrices, node_labels, graph_labels
# # 创建数据集实例
# dataset = CustomDataset(adjacency_matrices, node_labels, graph_labels)

# # 使用 DataLoader 加载数据
# data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)


# 定义GCN模型，主要用来两层卷积一层全连接层，将每张图进行二分类
class GCN(torch.nn.Module):
    def __init__(self,input_dim, output_dim=2):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv2d(1, kernel_size=4,out_channels= 16,stride=2,padding=1,bias=True)
        self.conv2 = nn.Conv2d(16, kernel_size=4,out_channels=32,stride=2,padding=1,bias=True)
        #这两个卷积暂时没用到，后续希望能加进去
        self.conv3=GCNConv(1,16)
        self.conv4=GCNConv(16,32)
        self.fc = nn.Linear(1568, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x=x.view(-1,28,28)
        # _,n_n,n_f=x.shape
        # x=torch.reshape(x, (188,1,n_n,n_f))
        # x=x.ravel()
        # edge_index=edge_index.coalesce()
        # new_sparse_matrix = torch.sparse_coo_tensor( values=edge_index.values(), indices=edge_index.indices(),sizes=(1, 28, 28))
        # temp=self.conv1(x, edge_index)

        temp=self.conv3(x, edge_index)
        x = F.relu(temp)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)
        # 展平特征图以匹配全连接层的输入
        x = x.view(x.size(0), -1)  # x.size(0)为batch_size
        x = torch.mean(x, dim=0, keepdim=True)
        # 应用全连接层
        x = self.fc(x)/28
        x = torch.sigmoid(x)

        x = x.view(-1, 1)
        return x

# 训练和评估
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(28*4).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)


# 生成从180到0的连续整数序列
sequence = list(range(188, -1, -1))
# 这两个损失函数都能用，区别不大
# criterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss()

# 随机打乱序列
random.shuffle(sequence)

# 按照120:60的比例分割序列
split_index = 120  # 计算分割点
array1 = sequence[:split_index]  # 第一个数组，包含120个元素
array2 = sequence[split_index:]   # 第二个数组，包含60个元素
los=[]
average_loss=[]
def train(train_loader,arry):
    model.train()
    # np.random.randiant(0,1)
    batch=[]
    #这一段注释掉的是使用多批次进行训练的，有些小问题，主要是batch是列表需要转换成张量用tensor.stack(list)应该能解决
    # for idx, batch in enumerate(train_loader):
    # for j in range(3):
    #     sample=random.sample(arry,batch_size)
    #     for i in range(len(sample)):
    #         batch.append(train_loader[i-1])
    # batch = [b.to(device) for b in batch] 
    temp_loss=0  
    for i in arry:
        batch=train_loader[i-1]
        batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        target=batch.y.view(-1,1)
        target.data=(target.data+1)/2
        # 计算损失
        loss = criterion(output, target)
        temp_loss+=loss
        los.append(loss)
        # loss = F.nll_loss(output, batch.y)
        loss.backward()
        optimizer.step()
    average_loss.append(temp_loss/len(arry))

def test(loader,arry):
    model.eval()
    correct = 0
    # for data in loader:
    # for j in range(3):
    #     sample = random.choices(arry,size=batch_size)
    #     for i in range(len(sample)):
    #         data=torch.stack(loader[i-1])
    acc=0
    for i in arry:
        data=loader[i-1]
        data = data.to(device)
        output = model(data)
        output = (output > 0.5).float()
        target=(data.y.view(-1,1)+1)/2
        target=target.view(-1,1)
        # pred = output.argmax(dim=1)

        acc+=accuracy_score(target,output)
        correct += torch.sum((output == target))
    return float(correct) / len(loader),acc/len(arry)

for epoch in range(200):
    random.shuffle(array1)
    random.shuffle(array2)#,进行再次打乱
    train(train_loader,array1)
    train_acc ,trainacc= test(train_loader,array1)
    test_acc,testacc = test(train_loader,array2)
    #返回计算结果中第二个为识别成功率，是现用指标
    # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    print(f'Epoch: {epoch:03d}, Train Acc: {trainacc:.4f}, Test Acc: {testacc:.4f}')

with open(r'C:\Users\86156\Desktop\program\python\graph\loss.txt','w') as f:
    for i in range(200):
        out =f'epoch: {i} , loss {average_loss[i]:2f} \n'
        f.write(out)

#现在最大的问题是sigmod似乎及其容易出现过拟合，很大概率会将数据判定为1，可以跟踪模型输出的output进行调参

