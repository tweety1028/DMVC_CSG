import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import v_measure_score, adjusted_rand_score, accuracy_score, fowlkes_mallows_score
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch_clustering.kmeans import PyTorchKMeans
from sklearn.cluster import KMeans

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def evaluate(label, pred):
    nmi = v_measure_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    f_score = fowlkes_mallows_score(label, pred)
    return nmi, ari, acc, pur, f_score

def inference(loader, model, data_size):
    model.eval()
    commonZ_list = []
    labels_vector = []
    for batch_idx, (xs, y, _) in enumerate(loader):
        with torch.no_grad():
            xrs, zs = model(xs)
            commonz = model.fusion(zs)
            commonZ_list.append(commonz)
        labels_vector.extend(y)
    commonZ = torch.cat(commonZ_list, dim=0)
    labels_vector = np.array(labels_vector).reshape(data_size)
    return labels_vector, commonZ

def inference2(loader, model, data_size, numview):
    model.eval()
    commonZ_list = []
    labels_vector = []
    for batch_idx, (xs, y, _) in enumerate(loader):
        with torch.no_grad():
            xrs, zs = model(xs)
            fused_z = torch.cat(zs, dim=1)
            commonZ_list.append(fused_z)
        labels_vector.extend(y)
    commonZ = torch.cat(commonZ_list, dim=0)
    labels_vector = np.array(labels_vector).reshape(data_size)
    return labels_vector, commonZ

def inference3(loader, model, data_size, numview):
    model.eval()
    commonZ_list = []
    labels_vector = []
    for batch_idx, (xs, y, _) in enumerate(loader):
        with torch.no_grad():
            xrs, _, _, enco_xs = model(xs)
            commonZ_list.append(enco_xs)
        labels_vector.extend(y)
    commonZ = torch.cat(commonZ_list, dim=0)
    labels_vector = np.array(labels_vector).reshape(data_size)
    return labels_vector, commonZ

def visualize_data_tsne(Z, labels, num_clusters, Dataname, epoch):
    matplotlib.use('Agg')
    labels = labels.astype(int)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    z_tsne = tsne.fit_transform(Z)
    fig = plt.figure()
    plt.scatter(z_tsne[:, 0], z_tsne[:, 1], s=2, c=labels, cmap=plt.cm.get_cmap("jet", num_clusters))
    plt.xticks([])
    plt.yticks([])
    plt.xlim(-105, 105)
    plt.ylim(-105, 105)
    plt.show()
    fig.savefig("/media/dell/1TB6/wxh/XXMVC - 副本/visualize/"+Dataname+str(epoch)+".png", dpi=300)
def valid(model, dataset, data_size, class_num):

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        drop_last=False,
    )
    labels_vector, commonZ = inference(test_loader, model, data_size)

    # print("Clustering results on low-level features of each view:")
    #
    # for v in range(4):
    #     y_pred = model.clustering(Zs[v])
    #     y_pred = y_pred.cpu().numpy()
    #     nmi, ari, acc, pur, f_score = evaluate(labels_vector, y_pred)
    #     print('ACC{} = {:.4f} NMI{} = {:.4f} ARI{} = {:.4f} PUR{}={:.4f}'.format(v + 1, acc,
    #                                                                              v + 1, nmi,
    #                                                                              v + 1, ari,
    #                                                                              v + 1, pur))

    print('Clustering results:')
    y_pred = model.clustering(commonZ)
    y_pred = y_pred.cpu().numpy()
    # kmeans = KMeans(n_clusters=class_num, n_init=100)
    # commonZ = commonZ.cpu()
    # y_pred = kmeans.fit_predict(commonZ)
    nmi, ari, acc, pur, f_score = evaluate(labels_vector, y_pred)
    print('ACC = {:.4f} NMI = {:.4f} PUR={:.4f} ARI = {:.4f} f_score = {:.4f}'.format(acc, nmi, pur, ari, f_score))
    return nmi, ari, acc, pur, f_score

def valid2(model, dataset, data_size, class_num, numview):

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        drop_last=False,
    )
    labels_vector, commonZ = inference2(test_loader, model, data_size, numview)
    print('Clustering results:')
    y_pred = model.clustering(commonZ)
    y_pred = y_pred.cpu().numpy()
    # kmeans = KMeans(n_clusters=class_num, n_init=100)
    # commonZ = commonZ.cpu()
    # y_pred = kmeans.fit_predict(commonZ)
    nmi, ari, acc, pur, f_score = evaluate(labels_vector, y_pred)
    print('ACC = {:.4f} NMI = {:.4f} PUR={:.4f} ARI = {:.4f} f_score = {:.4f}'.format(acc, nmi, pur, ari, f_score))

def valid3(model, dataset, data_size, class_num, numview):

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        drop_last=False,
    )
    labels_vector, commonZ = inference3(test_loader, model, data_size, numview)
    print('Clustering results:')
    y_pred = model.clustering(commonZ)
    y_pred = y_pred.cpu().numpy()
    # kmeans = KMeans(n_clusters=class_num, n_init=100)
    # commonZ = commonZ.cpu()
    # y_pred = kmeans.fit_predict(commonZ)
    nmi, ari, acc, pur, f_score = evaluate(labels_vector, y_pred)
    print('ACC = {:.4f} NMI = {:.4f} PUR={:.4f} ARI = {:.4f} f_score = {:.4f}'.format(acc, nmi, pur, ari, f_score))