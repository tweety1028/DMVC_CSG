import time

import numpy as np
import torch
from dataloader import *
import argparse
import torch.nn.functional as F
import torch.nn as nn
#
# a = np.array([1, 2, 3]).reshape(1, 3)
# X = np.array([[1, 5, 4,],
#              [4, 8, 6],
#              [1, 7, 9]]).reshape(3, 3)
# a = torch.from_numpy(a)
# X = torch.from_numpy(X)
# diff = torch.abs(X.unsqueeze(1) - X.unsqueeze(0))
# print(diff)
# diff = diff.permute(0, 2, 1)
# print(diff)
# weighted_diff = a
# print(weighted_diff)
# weighted_diff = weighted_diff.squeeze(1)
# sigma_result = torch.relu(weighted_diff)
# denominator = torch.exp(sigma_result).sum(dim=1)
# S = torch.exp(sigma_result) / denominator.view(-1, 1)
# print(S)
# clustering_model = PyTorchKMeans(metric='cosine',
#                                  init='k-means++',
#                                  random_state=0,
#                                  n_clusters=1000,
#                                  n_init=10,
#                                  max_iter=300,
#                                  tol=1e-4,
#                                  distributed=False,
#                                  verbose=True)
# X = torch.randn(60000, 256).cuda()
# clustering_model.fit_predict(X)

# def get_knn_graph(data, k):
#     num_samples = data.size(0)
#     graph = torch.zeros(num_samples, num_samples, dtype=torch.float32)
#
#     for i in range(num_samples):
#         distance = torch.sum((data - data[i])**2, dim=1)
#         _, small_indices = torch.topk(distance, k, largest=False)  # +1 to exclude self from neighbors
#         # Fill 1 in the graph for the k nearest neighbors
#         graph[i, small_indices[1:]] = 1
#
#     # Ensure the graph is symmetric
#     result_graph = torch.max(graph, graph.t())
#
#     return result_graph
#
# torch.cuda.set_device(0)
# Dataname = 'MNIST-USPS'
# parser = argparse.ArgumentParser(description='train')
# parser.add_argument('--dataset', default=Dataname)
# args = parser.parse_args()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# mv_data = MultiviewData(args.dataset, device)
# mv_data_loader, num_views, num_samples, _ = get_all_multiview_data(mv_data)
# start = time.time()
# for batch_idx, (sub_data_views, _, _) in enumerate(mv_data_loader):
#         result_graph = get_knn_graph(sub_data_views[1], 5)
# print(time.time()-start)
# print(result_graph)

# path = "D:\学习资料\datasets"
# mat = sio.loadmat(os.path.join(path, 'MNIST_USPS.mat'))
# X1 = mat['X1'].astype(np.float32)
# num_samples = X1.shape[0]
# dists = np.zeros((num_samples, num_samples))
# start = time.time()
# for i in range(num_samples):
#     dists[i] = np.sum(np.square(X1- X1[i]), axis=1).T
# print(dists)
# print(time.time() - start)
#
# result_graph = result_graph.cpu().numpy()
# graph = np.zeros((num_samples,num_samples),dtype = np.int)
# for i in range(num_samples):
#     distance = dists[i,:]
#     small_index = np.argsort(distance)
#     graph[i,small_index[0:5]] = 1
# graph = graph-np.diag(np.diag(graph))
# resultgraph = np.maximum(graph,np.transpose(graph))
# print(resultgraph)
#
# are_equal = np.array_equal(resultgraph, result_graph)
#
# if are_equal:
#     print("两个数组相等")
# else:
#     print("两个数组不相等")




# import torch
#
# # 假设 labels 是包含 n 个样本的标签的张量
# labels = torch.tensor([0, 1, 0, 2, 2, 1])  # 这里是一个示例标签张量
#
# # 计算 n*n 的匹配矩阵
# n = len(labels)
# start = time.time()
# matching_matrix = (labels.view(-1, 1) == labels.view(1, -1)).int()
#
# # 打印结果
# print(matching_matrix)
# print(time.time() - start)
#
# import numpy as np
#
# # 假设 labels 是包含 n 个样本的标签的向量
# labels = np.array([0, 1, 0, 2, 2, 1])  # 这里是一个示例标签向量
#
# # 计算 n*n 的匹配矩阵
# n = len(labels)
# start = time.time()
# matching_matrix = (labels[:, np.newaxis] == labels[np.newaxis, :]).astype(int)
# print(time.time() - start)
# # 打印结果
# print(matching_matrix)
#
#
# matching_matrix = np.zeros((n, n))
# start = time.time()
# for i in range(n):
#     for j in range(n):
#         if labels[i] == labels[j]:
#             matching_matrix[i, j] = 1
#
# # 打印结果
# print(time.time()-start)
# print(matching_matrix)

# path = "D:\学习资料\datasets"
# mat = sio.loadmat(os.path.join(path, 'MNIST_USPS.mat'))
# labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
# n = len(labels)
# start = time.time()
# matching_matrix = (labels[:, np.newaxis] == labels[np.newaxis, :]).astype(int)
# print(time.time() - start)
# # 打印结果
# print(matching_matrix)

# zi = torch.from_numpy(np.array([[0.2, 0.1, 1.5],
#                [0.3, 0.9, 0.1]]))
#
# z = torch.from_numpy(np.array([[0.15, 0.2, 1.3],
#                [0.45, 0.7, 0.25]]))
#
# w = torch.from_numpy(np.array([[1, 1],
#                [1, 1]]))
#
# y_pse = torch.from_numpy(np.array([[1, 1],
#                [1, 1]]))
# cross_view_distance = zi @ z.t()
# print(cross_view_distance)
# positive_loss = (w & y_pse) * cross_view_distance
# print(positive_loss.sum())

# c = []
#
# a = torch.tensor([[0.1, 0.2, 0.5],
#                   [0.2, 0.3, 0.1]])
# b = torch.tensor([[0.12, 0.52, 0.45],
#                   [0.21, 0.13, 0.51]])
#
# c.append(a)
# c.append(b)
# print(torch.sum(torch.stack(c), dim=0))
a = torch.tensor([-0.2, 0.4, 1.2])
w = torch.sigmoid(a)
print(w)
w = w / torch.sum(w)
print(w)