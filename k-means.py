import numpy as np
from sklearn import datasets


def calc_eclud_dist(a, b):  # 计算欧氏距离
    return np.sqrt(np.sum(np.power(a - b, 2)))


def rand_centroid_pick(data, k):  # 随机抽取中心点
    n = np.shape(data)[0]
    rand_centroid_row = np.arange(n)
    np.random.shuffle(rand_centroid_row)
    rand_centroid = data[rand_centroid_row[0:3]]
    return rand_centroid


def update_centroid(cluster_dict, k, n):  # 产生新的中心点
    centroid_set = np.zeros((k, n))
    i = 0
    for key in cluster_dict.keys():
        if cluster_dict[key] is not None:
            cluster = np.array(cluster_dict[key])
            centroid = np.reshape(np.mean(cluster, axis=0), (1, 4))
            centroid_set[i] = centroid
            i += 1
    return centroid_set


def k_means(k, data):  # k-means算法
    cluster_dict = {}  # 存储每类的样本
    n = np.shape(data)[1]
    dist = np.zeros(k)  # 存储距离
    iter = 0  # 迭代次数
    cluster_changed = True
    centroid_set = rand_centroid_pick(data, k)
    while cluster_changed:
        for g in range(k):
            cluster_dict.setdefault(f'G{g}')
        for d in data:
            d = np.reshape(d, (1, 4))
            for i in range(k):
                dist[i] = calc_eclud_dist(d, centroid_set[i])
            for j in range(k):
                if dist[j] == np.min(dist):
                    if cluster_dict[f'G{j}'] is None:
                        cluster_dict[f'G{j}'] = d
                    else:
                        cluster_dict[f'G{j}'] = np.append(cluster_dict[f'G{j}'], d, axis=0)

            dist = np.zeros(k)
        old_centroid_set = centroid_set
        centroid_set = update_centroid(cluster_dict, k, n)
        iter += 1
        if (old_centroid_set == centroid_set).all():
            cluster_changed = False
        else:
            cluster_dict = {}
        print(f'iter:{iter}')
    return iter, cluster_dict


if __name__ == '__main__':
    iris = datasets.load_iris()
    k = 3
    iter, cluster = k_means(k, iris.data)
    print(f'第{iter}次迭代，迭代完毕')
    for k in cluster.keys():
        print(f'\n{k}:{cluster[k]}')