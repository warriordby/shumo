from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


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
        ax.scatter(X[i, 0], X[i, 1], X[i, 2], color=colors[y[i]], s=1, label=y[i] if i == 0 else "")  # 只为第一个添加标签

    # 添加图例
    unique_labels = np.unique(y)
    for idx, label in enumerate(unique_labels):
        ax.scatter([], [], [], color=colors[idx], label=str(label), s=1)

    ax.legend(title="Labels")
    
    if title is not None:
        plt.title(title)

    plt.savefig(filename)  # 保存图像
    plt.show()  # 显示图像
    plt.close()  # 关闭当前图形以释放内存


# 计算调整后的坐标
def adjust_coordinates(X, y_kmeans, centroids):
    adjusted_X = np.copy(X)
    for i in range(len(centroids)):
        cluster_points = X[y_kmeans == i]
        centroid = centroids[i]
        for point in cluster_points:
            distance = np.linalg.norm(point - centroid)
            # 调整点的位置，使其更靠近质心
            adjusted_X[y_kmeans == i] += (centroid - point) * 0.1 / (distance + 1e-5)  # 防止除零
    return adjusted_X


def plot_centroid_visualization(X, y_kmeans, centroids, title=None, filename='centroid_visualization.png'):
    """Plot cluster centroids with size based on maximum distance to points in the cluster."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 计算每个聚类的点到聚类中心的最大距离
    max_distances = []
    for i in range(len(centroids)):
        cluster_points = X[y_kmeans == i]
        if len(cluster_points) > 0:
            centroid = centroids[i]
            max_distance = np.max(np.linalg.norm(cluster_points - centroid, axis=1))
            max_distances.append(max_distance)
        else:
            max_distances.append(0)

    # 创建颜色映射
    unique_labels = np.unique(y_kmeans)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

    # 为每个类别绘制不同颜色的点
    for idx, label in enumerate(unique_labels):
        ax.scatter(X[y_kmeans == label, 0], X[y_kmeans == label, 1], X[y_kmeans == label, 2], 
                   color=colors[idx], s=1, alpha=0.8, label=f'Cluster {label}')

    # 绘制聚类中心，颜色与类别相同，大小基于最大距离
    for i in range(len(centroids)):
        ax.scatter(centroids[i, 0], centroids[i, 1], centroids[i, 2], 
                   s=max_distances[i] * 500,  # Scale size
                   color=colors[i], alpha=0.5, label=f'Centroid {i}')
    # 创建自定义图例句柄
    legend_handles = []
    for i in range(len(centroids)):
        legend_handles.append(ax.scatter([], [], color=colors[i], s=50, alpha=0.8, label=f'Centroid {i}'))

    # 添加标题和图例
    if title is not None:
        plt.title(title)

    ax.legend(handles=legend_handles + 
              [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {label}', 
                           markerfacecolor=colors[idx], markersize=5) for idx, label in enumerate(unique_labels)], 
              loc='upper right', bbox_to_anchor=(1.35, 1))
    
    plt.savefig(filename)  # 保存图像
    plt.show()  # 显示图像
    plt.close()  # 关闭当前图形以释放内存


if __name__=='__main__':

    # 图像文件名和路径
    img_list = ['1_1', '9_9','5_5','1_8','5_9']
    root_path = r"F:\数据集\LFsyn\data\UrbanLF-Real\train\Image109"
    images = []

    # 读取图像并重塑为数组
    for img_name in img_list:
        path = os.path.join(root_path, img_name + '.png')
        img = Image.open(path)
        img = img.resize((32, 32), Image.LANCZOS)
        images.append(np.array(img).reshape(-1, 3))

    # 拼接图像数组
    X = np.concatenate(images, axis=0)

    # 设计标签 y
    y = np.array([i for i in range(len(img_list)) for _ in range(images[i].shape[0])])

    # 查找是否存在相同位置的像素点
    new_label = len(img_list)  # 新类别的标签
    pixel_labels = np.copy(y)  # 初始化标签

    # 遍历每个像素位置
    for idx in range(X.shape[0]):
        current_pixel = X[idx]

        # 检查该像素在所有类别中的值是否相同
        same_pixel = True
        for i in range(len(img_list)):
            # 获取当前类别的像素值
            category_pixels = X[y == i]
            if idx < category_pixels.shape[0]:
                current_category_pixel = category_pixels[idx]
                if not np.array_equal(current_pixel, current_category_pixel):
                    same_pixel = False
                    break
            else:
                same_pixel = False
                break

        # 如果所有类别的像素值相同，修改标签
        if same_pixel:
            pixel_labels[y == y[idx]] = new_label

    # 转换为数组
    y = pixel_labels
    # 打印形状以确认
    print(X.shape)

    print("Computing t-SNE embedding")
    tsne3d = TSNE(n_components=3, init='pca', random_state=0)

    X_tsne_3d = tsne3d.fit_transform(X)


    from sklearn.cluster import KMeans

    # K-means 聚类
    n_clusters = len(np.unique(y))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    y_kmeans = kmeans.fit_predict(X_tsne_3d)

    # 计算聚类中心
    centroids = kmeans.cluster_centers_

    # 在生成 t-SNE 之后，调整坐标
    adjusted_X_tsne_3d = adjust_coordinates(X_tsne_3d, y_kmeans, centroids)


    # plot_embedding_3d(X_tsne_3d[:, 0:3], y, "t-SNE 3D", "embedding_3d.png")

    # 使用调整后的坐标绘制 3D 图
    # plot_embedding_3d(adjusted_X_tsne_3d, y_kmeans, "Adjusted t-SNE 3D", "adjusted_embedding_3d.png")

    # 调用新函数以绘制聚类中心
    plot_centroid_visualization(X_tsne_3d, y_kmeans, centroids, "Cluster Centroids with Distances", "centroid_visualization.png")
