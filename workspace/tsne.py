from sklearn import datasets
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# 图像文件名和路径
img_list = ['1_1', '9_9']
root_path = r"F:\数据集\LFsyn\data\UrbanLF-Real\train\Image109"
images = []

# 读取图像并重塑为数组
for img_name in img_list:
    path = os.path.join(root_path, img_name + '.png')
    img = Image.open(path)
    img = img.resize((16, 16), Image.LANCZOS)
    images.append(np.array(img).reshape(-1, 3))

# 拼接图像数组
X = np.concatenate(images, axis=0)

# 设计标签 y
y = np.array([i for i in range(len(img_list)) for _ in range(images[i].shape[0])])

# 查找像素值完全相同的点
unique_pixels = np.unique(X, axis=0)

# 处理标签以包含额外类别
extra_label = len(img_list)  # 新类别的标签
pixel_labels = []

# for pixel in X:
#     if np.sum(np.all(X == pixel, axis=1)) > 1:  # 检查是否有相同的像素
#         pixel_labels.append(extra_label)
#     else:
#         pixel_labels.append(y[np.where((X == pixel).all(axis=1))[0][0]])

# 转换为数组
# y = np.array(pixel_labels)

# 打印形状以确认
print(X.shape)

from mpl_toolkits.mplot3d import Axes3D

def plot_embedding_2d(X, y, title=None, filename='embedding_2d.png'):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # 绘制 2D 图形
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set3(y[i] / 5.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    
    plt.savefig(filename)  # 保存图像
    plt.close()  # 关闭当前图形以释放内存

def plot_embedding_3d_text(X, y, title=None, filename='embedding_3d.png'):
    """Plot a 3D embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
    
    # 创建 3D 图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], X[i, 2], str(y[i]),
                color=plt.cm.Set3(y[i] / 5.),
                fontdict={'weight': 'bold', 'size': 9})

    if title is not None:
        plt.title(title)
    plt.show()
    plt.savefig(filename)  # 保存图像
    plt.close()  # 关闭当前图形以释放内存

def plot_embedding_3d(X, y, title=None, filename='embedding_3d.png'):
    """Plot a 3D embedding X with class labels as spheres and colors as legend."""
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)

    # 创建 3D 图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 创建颜色映射
    colors = plt.cm.Set3(np.linspace(0, 1, len(np.unique(y))))

    # 绘制每个点为球体
    for i in range(X.shape[0]):
        ax.scatter(X[i, 0], X[i, 1], X[i, 2], color=colors[y[i]], s=100, label=y[i] if i == 0 else "")  # 只为第一个添加标签

    # 添加图例
    unique_labels = np.unique(y)
    for idx, label in enumerate(unique_labels):
        ax.scatter([], [], [], color=colors[idx], label=str(label), s=100)

    ax.legend(title="Labels")
    
    if title is not None:
        plt.title(title)

    plt.savefig(filename)  # 保存图像
    plt.show()  # 显示图像
    plt.close()  # 关闭当前图形以释放内存

print("Computing t-SNE embedding")
tsne2d = TSNE(n_components=2, init='pca', random_state=0)
tsne3d = TSNE(n_components=3, init='pca', random_state=0)

# X_tsne_2d = tsne2d.fit_transform(X)
X_tsne_3d = tsne3d.fit_transform(X)

# plot_embedding_2d(X_tsne_2d[:, 0:2], y, "t-SNE 2D", "embedding_2d.png")
plot_embedding_3d(X_tsne_3d[:, 0:3], y, "t-SNE 3D", "embedding_3d.png")


from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# K-means 聚类
n_clusters = len(np.unique(y))
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
y_kmeans = kmeans.fit_predict(X_tsne_3d)

# 计算聚类中心
centroids = kmeans.cluster_centers_

# 计算类内距离
intra_cluster_distances = []
for i in range(n_clusters):
    cluster_points = X_tsne_3d[y_kmeans == i]
    centroid = centroids[i]
    intra_cluster_distance = np.mean(np.linalg.norm(cluster_points - centroid, axis=1))
    intra_cluster_distances.append(intra_cluster_distance)

# 计算类间距离
inter_cluster_distances = pairwise_distances(centroids)



# 计算总的目标函数值
objective_function = np.sum(intra_cluster_distances) - np.sum(inter_cluster_distances)


# 使用 PCA 将距离投影到二维
pca = PCA(n_components=2)
# 进行降维
X_pca = pca.fit_transform(X_tsne_3d)

# 可视化最终结果
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', s=50)

# 添加图例
legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
plt.gca().add_artist(legend1)

plt.title('2D Projection with Maximum Discriminability')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter)
plt.show()