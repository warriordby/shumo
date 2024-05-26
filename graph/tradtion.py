import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from traditional_model import model_list
from sklearn.metrics import accuracy_score
import networkx as nx


# 加载数据
data = scipy.io.loadmat(r"C:\Users\86156\Downloads\2024年湖南大学数学建模竞赛赛题及参赛须知\2024年湖南大学数学建模竞赛赛题及参赛须知\B题：药物属性预测\附件：MMM_data.mat")
am = data['MMM']['am'][0]  # 邻接矩阵
al = data['MMM']['al'][0]  # 节点标签
nl = data['MMM']['nl'][0]  # 图标签
el = data['MMM']['el']  # 图的其他标签
label=data['lable']
t=am[0][0][0].data
# 特征提取函数
def extract_features(am):
    features = []
    for i in range(len(am)):
        G = nx.from_numpy_matrix(am[i])
        degree = np.mean([d for n, d in G.degree()])
        clustering_coefficient = nx.average_clustering(G)#平均聚类
        features.append([degree, clustering_coefficient])
    return np.array(features)

X = extract_features(am)
y = label.ravel()

#将am的lable展开
def process_label(y):
    graph_len=y.shape[0]
    new_label = []
    # 处理y的最后一个维度，并将处理后的结果赋值给新数组的相应位置
    for i in range(graph_len):

        node_len=y[i].shape[0]
        temp_node=[]
        for j in range(node_len):
            temp_label=[0 for _ in range(node_len)]
            label = y[i][j] 
            for k in label[0]:
                for val in k:
                    temp_label[val-1]=1
            temp_node.append(temp_label)  # 赋值给新数组的相应位置
        new_label.append(temp_node)
    return np.array(new_label)
# y=process_label(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
for model in model_list:
    # 训练模型
    model.fit(X_train, y_train)
    # 预测
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Classification accuracy using {model.__class__.__name__}: {accuracy:.2f}')