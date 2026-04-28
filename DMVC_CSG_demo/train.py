import matplotlib.pyplot as plt

from network import Network
from model import *
import numpy as np
import argparse
import random
import os
from loss import *

Dataname = 'Wiki'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--load_model', default=False, help='Testing if True or training.')
parser.add_argument('--save_model', default=False, help='Saving the model after training.')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=0.5)
parser.add_argument("--learning_rate", default=0.0001)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--mse_epochs", default=200)
parser.add_argument("--con_epochs", default=500)
parser.add_argument("--feature_dim", default=256)
parser.add_argument("--large_datasets", default=False, type=str)
parser.add_argument("--k", default=5)
parser.add_argument('--fusion_way', default='softmax', type=str)
parser.add_argument('--gpu', default='0', type=str, help='GPU device idx.')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def vis(epoch):
    mv_data_loader, _, _, _ = get_all_multiview_data(mv_data)

    zs_list = []
    ys_list = []
    with torch.no_grad():
        for batch_idx, (sub_data_views, y, _) in enumerate(mv_data_loader):
            _, z = network(sub_data_views)
            z = network.fusion(z)
            zs_list.append(z)
            ys_list.append(y)
        zs = torch.cat(zs_list, dim=0)
        ys = torch.cat(ys_list, dim=0)
        zs = zs.cpu().numpy()
        ys = ys.cpu().numpy()
    visualize_data_tsne(zs, ys, num_clusters, Dataname, epoch)

if __name__ == "__main__":
    if args.dataset == 'Wiki':
        args.learning_rate = 0.0001
        args.batch_size = 256
        args.seed = 10
        args.con_epochs = 200
        args.k = 7
        lmd = 0.01
        beta = 1e-05

    for lmd in [0.01]:
      for beta in [1e-05]:
        # print("==========\nArgs:{}\n==========".format(args))
        print(str(lmd) + "\n")
        set_seed(args.seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mv_data = MultiviewData(args.dataset, device)
        num_views = len(mv_data.data_views)
        num_samples = mv_data.labels.size
        num_clusters = np.unique(mv_data.labels).size
        input_sizes = np.zeros(num_views, dtype=int)
        for idx in range(num_views):
            input_sizes[idx] = mv_data.data_views[idx].shape[1]

        network = Network(num_views, num_samples, num_clusters, input_sizes, args.feature_dim, args.fusion_way)
        network = network.to(device)
        optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        mvc_loss = Loss(args.batch_size, num_clusters, args.temperature_l, args.temperature_f).to(device)
        if args.load_model:
            state_dict = torch.load('./models/pytorch_model_%s.pth' % args.dataset)
            network.load_state_dict(state_dict)
        else:
            epoch_list = []
            totalloss_list = []
            t = time.time()
            pre_train_loss_values = pre_train(network, mv_data, args.batch_size, args.mse_epochs, optimizer)
            valid(network, mv_data, num_samples, num_clusters)
            # vis(0)
            if args.large_datasets == False:
                for epoch in range(1, args.con_epochs + 1):
                    total_loss = contrastive_train(network, mv_data, mvc_loss, args.batch_size, epoch, lmd, beta, optimizer, args.k)
                    epoch_list.append(epoch)
                    totalloss_list.append(total_loss)
                nmi, ari, acc, pur, f_score = valid(network, mv_data, num_samples, num_clusters)
                print(network.get_weights("softmax"))
            else:
                for epoch in range(1, args.con_epochs + 1):
                    total_loss = contrastive_train(network, mv_data, mvc_loss, args.batch_size, epoch, lmd, beta, optimizer, args.k)
                    epoch_list.append(epoch)
                    totalloss_list.append(total_loss)
                nmi, ari, acc, pur, f_score = valid(network, mv_data, num_samples, num_clusters)
                print(network.get_weights("softmax"))

        # print("Time elapsed: {:.4f}s".format(time.time() - t))
        # fig = plt.figure()
        # x_t = range(0, 101, 10)
        # plt.xticks(x_t)
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss value')
        # plt.plot(epoch_list, totalloss_list)
        # plt.show()
        # fig.savefig("/media/dell/1TB1/lyt/mutual_demo/lossvalue/" + Dataname + ".png", dpi=fig.dpi)