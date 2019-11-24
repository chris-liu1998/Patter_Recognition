import numpy as np


def load_data():
    w1 = np.array([[0, 0, 0],
                   [1, 0, 0],
                   [1, 0, 1],
                   [1, 1, 0]], dtype=float)
    w2 = np.array([[0, 0, 1],
                   [1, 1, 0],
                   [0, 1, 1],
                   [1, 1, 1]], dtype=float)
    return w1, w2


def mean_vector(w1, w2):
    x1_bar = np.reshape(np.mean(w1, axis=0), (1, w1.shape[1]))
    x2_bar = np.reshape(np.mean(w2, axis=0), (1, w2.shape[1]))
    return x1_bar, x2_bar


def change_coord(w1, w2):  # 改变坐标系
    x1_bar, x2_bar = mean_vector(w1, w2)
    for x in w1:
        x = np.reshape(x, (1, w1.shape[1]))
        x -= x1_bar
    for x in w2:
        x = np.reshape(x, (1, w2.shape[1]))
        x -= x2_bar
    return w1, w2


def get_autoc_mat(w1, w2):  # 计算自相关矩阵
    r1 = np.zeros((w1.shape[1], w1.shape[1]))
    r2 = np.zeros((w2.shape[1], w2.shape[1]))
    for x in w1:
        x = np.reshape(x, (1, w1.shape[1]))
        r1 += np.dot(x.T, x)
    for x in w2:
        x = np.reshape(x, (1, w2.shape[1]))
        r2 += np.dot(x.T, x)
    r1 /= len(w1)
    r2 /= len(w2)
    return r1, r2


def reduct_dim(d, w1, w2):  # 降维主函数
    new_w1, new_w2 = change_coord(w1, w2)
    r1, r2 = get_autoc_mat(new_w1, new_w2)
    print(r1,r2)
    eval1, evec1 = np.linalg.eig(r1)
    sorted_indices1 = np.argsort(-eval1)
    top_vec1 = evec1[:, sorted_indices1[0:d]]
    eval2, evec2 = np.linalg.eig(r2)
    sorted_indices2 = np.argsort(-eval2)
    top_vec2 = evec2[:, sorted_indices2[0:d]]
    y1 = np.dot(w1, top_vec1)
    y2 = np.dot(w2, top_vec2)
    return y1, y2


if __name__ == '__main__':
    w1, w2 = load_data()
    d = 1
    print(reduct_dim(d, w1, w2))
    d = 2
    print(reduct_dim(d, w1, w2))
