import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, batch_size, num_clusters, temperature_l, temperature_f):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.num_clusters = num_clusters
        self.temperature_l = temperature_l
        self.temperature_f = temperature_f
        self.similarity = nn.CosineSimilarity(dim=2)
        self.lambda_var = 0.4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def compute_cluster_loss(self, clusters, h1, centers, label, h2, k):

        similarity_matrix = torch.exp(self.similarity(h1.unsqueeze(1), h2.unsqueeze(0)) / self.temperature_l)
        # 取前 k 个最相似的样本（不包括自身）
        topk_values, topk_indices = torch.topk(similarity_matrix, k=k, dim=1)  # (N, k+1)
        # 去掉自身（第一个元素是自身）
        sim_positive = topk_values[:, :]   # (N, k)
        # 计算负样本
        indicator = torch.ones(h1.size(0), clusters).to(h1.device)
        for i in range(h1.size(0)):
            indicator[i, label[i]] = 0  # 使自身所属的聚类中心不作为负样本

        sim_negative = torch.mul(
            torch.exp(self.similarity(h1.unsqueeze(1), centers.unsqueeze(0)) / self.temperature_l), indicator)

        loss = - torch.log(torch.sum(sim_positive, dim=1) /
                           (torch.sum(sim_positive, dim=1) + torch.sum(sim_negative, dim=1)))

        return loss.mean()

    def feature_loss(self, q_centers):
        M = torch.mm(q_centers, q_centers.t())  # [K, K]

        # Subtract 2I to ignore self-similarity
        I = torch.eye(q_centers.size(0), device=q_centers.device)  # [K, K]
        M = M - 2 * I  # Diagonal becomes -1

        # For each prototype, find max similarity (excluding itself)
        max_sim, _ = M.max(dim=1)  # [K]

        # Average over all prototypes
        loss = max_sim.mean()

        return loss

    def feature_loss2(self, q_centers, k_centers):
        M = torch.mm(q_centers, q_centers.t())
        M2 = torch.mm(k_centers, k_centers.t())
        I = torch.eye(k_centers.size(0), device=k_centers.device)
        M2_I = M2 - 2 * I
        max_sim2, _ = M2_I.max(dim=1)
        log_q = F.log_softmax(M2 / 0.1, dim=1)
        p_k = F.softmax(M / 0.1, dim=1)
        kl_loss = F.kl_div(log_q, p_k, reduction='batchmean')
        loss = max_sim2.mean() + kl_loss

        return loss