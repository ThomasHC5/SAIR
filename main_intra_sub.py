import os

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor
import argparse
import time
import datetime
import numpy as np
import random
import itertools
import pandas as pd

from dataset.eegdataset_intra_sub import load_data

result_path = 'result/intra/'
model_path = 'model/intra/'
model_idx = 'intra_'

parser = argparse.ArgumentParser(description='Experiment Stimuli Recognition test with CLIP encoder')
parser.add_argument('--epoch', default=200, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--num_sub', default=10, type=int,
                    help='number of subjects used in the experiments. ')
parser.add_argument('-batch_size', '--batch-size', default=1000, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--device', default='cuda:0', type=str, help='device to use for training / testing. ')

args = parser.parse_args()

result_path = result_path + 'seed%d/' % (args.seed)
model_path = model_path + 'seed%d/' % (args.seed)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1 and hasattr(m, 'weight'):
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1 and hasattr(m, 'weight'):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('GATConv') != -1 and hasattr(m, 'lin'):
        weights_init_normal(m.lin)

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # revised from shallownet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = self.tsconv(x)
        x = self.projection(x)
        return x

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x

from torch_geometric.nn import GATConv
class EEG_GAT(nn.Module):
    def __init__(self, in_channels=250, out_channels=250):
        super(EEG_GAT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = GATConv(in_channels=in_channels, out_channels=out_channels, heads=1)

        self.num_channels = 64
        self.edge_index_list = torch.Tensor([(i, j) for i in range(self.num_channels) for j in range(self.num_channels) if i != j]).to(args.device)
        self.edge_index = torch.tensor(self.edge_index_list, dtype=torch.long).t().contiguous().to(args.device)

    def forward(self, x):
        batch_size, _, num_channels, num_features = x.size()
        x = x.view(batch_size*num_channels, num_features)
        x = self.conv1(x, self.edge_index)
        x = x.view(batch_size, num_channels, -1)
        x = x.unsqueeze(1)
        
        return x

class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    EEG_GAT(),
                    nn.Dropout(0.3),
                )
            ),
            PatchEmbedding(emb_size),
            FlattenHead()
        )

class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.2):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=1024, proj_dim=1024, drop_proj=0.2):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

class FiLMLayer(nn.Module):
    """Applies FiLM modulation."""
    def forward(self, x, gamma, beta):
        return x * gamma + beta + x

class SAIR(nn.Module):
    def __init__(self, eeg_channels: int, time_len: int,
                 clip_dim: int = 1024, heads: int = 3):
        super().__init__()
        self.encoder_edge = Enc_eeg()
        self.encoder_depth = Enc_eeg()
        self.encoder_scene   = Enc_eeg()
        self.proj_edge = Proj_eeg(proj_dim=clip_dim)
        self.proj_depth = Proj_eeg(proj_dim=clip_dim)
        self.proj_scene   = Proj_eeg(proj_dim=clip_dim)

        # --- FiLM Layers ---
        feature_dim = 1024  # The dimension of features from the router
        self.film_layer = FiLMLayer()

        self.film_gen_e_to_s = nn.Linear(feature_dim * 2, feature_dim * 2)

    def forward(self, eeg): # [1000, 1, 63, 250]
        ye = self.encoder_edge(eeg)
        yd = self.encoder_depth(eeg)
        ys = self.encoder_scene(eeg)

        ye_final = self.proj_edge(ye)
        yd_final = self.proj_depth(yd)
        ys_proj = self.proj_scene(ys)

        y = torch.cat([ye_final, yd_final], dim=-1)

        gamma_e_to_s, beta_e_to_s = self.film_gen_e_to_s(y).chunk(2, dim=-1)
        ys_modulated = self.film_layer(ys_proj, gamma_e_to_s, beta_e_to_s) + ys_proj

        return ye_final, yd_final, ys_modulated
    
class IMG_Proj(nn.Module):
    def __init__(self, clip_dim: int = 1024):
        super(IMG_Proj, self).__init__()
        self.proj_edge = Proj_img(proj_dim=clip_dim)
        self.proj_depth = Proj_img(proj_dim=clip_dim)
        self.proj_scene   = Proj_img(proj_dim=clip_dim)
    def forward(self, xe, xd, xs):
        xe = self.proj_edge(xe)
        xd = self.proj_depth(xd)
        xs = self.proj_scene(xs)
        return xe, xd, xs
    
class IE():
    def __init__(self, args, nsub):
        super(IE, self).__init__()
        self.args = args
        self.num_class = 200
        self.batch_size = args.batch_size
        self.batch_size_test = 400
        self.n_epochs = args.epoch
        self.lr = args.lr

        self.b1 = 0.9
        self.b2 = 0.999
        self.nSub = nsub

        self.start_epoch = 0
        self.eeg_data_path = 'Data/Things-EEG2/Preprocessed_data_250Hz/'
        self.img_data_path = 'Data/Things-EEG2/Features/'
        self.pretrain = False

        self.log_write = open(result_path + "log_subject%d.txt" % self.nSub, "w")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_cls = torch.nn.CrossEntropyLoss().to(args.device)
        self.Enc_eeg = SAIR(63, 250).to(args.device)
        self.Proj_imgf = IMG_Proj().to(args.device)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.centers = {}
        print('initial define done.')
    
    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):
        sub_model_idx = model_idx + 'Sub%d_' % (self.nSub)
        
        self.Enc_eeg.apply(weights_init_normal)
        self.Proj_imgf.apply(weights_init_normal)

        self.dataloader, self.val_dataloader, self.test_dataloader, test_image_edge, test_image_depth, test_image_scene = load_data(self.eeg_data_path, self.img_data_path, self.nSub, self.args.seed, self.batch_size, self.batch_size_test)

        # Optimizers
        self.optimizer = torch.optim.Adam(itertools.chain(self.Enc_eeg.parameters(), self.Proj_imgf.parameters()), lr=self.lr, betas=(self.b1, self.b2))

        best_loss_val = np.inf

        for e in range(self.n_epochs):

            self.Enc_eeg.train()
            self.Proj_imgf.train()

            for i, (eeg, img_edge, img_depth, img_scene) in enumerate(self.dataloader):

                eeg = Variable(eeg.to(self.args.device).type(self.Tensor))
                img_edge = Variable(img_edge.to(self.args.device).type(self.Tensor))
                img_depth = Variable(img_depth.to(self.args.device).type(self.Tensor))
                img_scene = Variable(img_scene.to(self.args.device).type(self.Tensor))
                labels = torch.arange(eeg.shape[0])  # used for the loss
                labels = Variable(labels.to(self.args.device).type(self.LongTensor))

                # obtain the features
                ye, yd, ys = self.Enc_eeg(eeg)
                eeg_features = torch.cat([ye, yd, ys], dim=1)
                ze, zd, zs = self.Proj_imgf(img_edge, img_depth, img_scene)
                img_features = torch.cat([ze, zd, zs], dim=1)

                # normalize the features
                ye = ye / ye.norm(dim=1, keepdim=True)
                yd = yd / yd.norm(dim=1, keepdim=True)
                ys = ys / ys.norm(dim=1, keepdim=True)
                ze = ze / ze.norm(dim=1, keepdim=True)
                zd = zd / zd.norm(dim=1, keepdim=True)
                zs = zs / zs.norm(dim=1, keepdim=True)
                eeg_features = eeg_features / eeg_features.norm(dim=1, keepdim=True)
                img_features = img_features / img_features.norm(dim=1, keepdim=True)

                # cosine similarity as the logits
                logit_scale = self.logit_scale.exp()
                logits_e_eeg = logit_scale * ye @ ze.t()
                logits_e_img = logits_e_eeg.t()
                logits_d_eeg = logit_scale * yd @ zd.t()
                logits_d_img = logits_d_eeg.t()
                logits_s_eeg = logit_scale * ys @ zs.t()
                logits_s_img = logits_s_eeg.t()
                logits_eeg = logit_scale * eeg_features @ img_features.t()
                logits_img = logits_eeg.t()

                loss_e_eeg = self.criterion_cls(logits_e_eeg, labels)
                loss_e_img = self.criterion_cls(logits_e_img, labels)
                loss_d_eeg = self.criterion_cls(logits_d_eeg, labels)
                loss_d_img = self.criterion_cls(logits_d_img, labels)
                loss_s_eeg = self.criterion_cls(logits_s_eeg, labels)
                loss_s_img = self.criterion_cls(logits_s_img, labels)
                loss_eeg = self.criterion_cls(logits_eeg, labels)
                loss_img = self.criterion_cls(logits_img, labels)

                loss = (loss_e_eeg + loss_e_img + loss_d_eeg + loss_d_img + loss_s_eeg + loss_s_img) / 6 + (loss_eeg + loss_img) / 2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (e + 1) % 1 == 0:
                self.Enc_eeg.eval()
                self.Proj_imgf.eval()
                with torch.no_grad():
                    # * validation part
                    for i, (veeg, vimg_edge, vimg_depth, vimg_scene) in enumerate(self.val_dataloader):

                        veeg = Variable(veeg.to(self.args.device).type(self.Tensor))
                        vimg_edge = Variable(vimg_edge.to(self.args.device).type(self.Tensor))
                        vimg_depth = Variable(vimg_depth.to(self.args.device).type(self.Tensor))
                        vimg_scene = Variable(vimg_scene.to(self.args.device).type(self.Tensor))
                        vlabels = torch.arange(veeg.shape[0])
                        vlabels = Variable(vlabels.to(self.args.device).type(self.LongTensor))

                        ye, yd, ys = self.Enc_eeg(veeg)
                        veeg_features = torch.cat([ye, yd, ys], dim=1)
                        ze, zd, zs = self.Proj_imgf(vimg_edge, vimg_depth, vimg_scene)
                        vimg_features = torch.cat([ze, zd, zs], dim=1)

                        veeg_features = veeg_features / veeg_features.norm(dim=1, keepdim=True)
                        vimg_features = vimg_features / vimg_features.norm(dim=1, keepdim=True)

                        logit_scale = self.logit_scale.exp()
                        vlogits_eeg = logit_scale * veeg_features @ vimg_features.t()
                        vlogits_img = vlogits_eeg.t()

                        vloss_eeg = self.criterion_cls(vlogits_eeg, vlabels)
                        vloss_img = self.criterion_cls(vlogits_img, vlabels)

                        vloss = (vloss_eeg + vloss_img) / 2

                        if vloss <= best_loss_val:
                            best_loss_val = vloss
                            best_epoch = e + 1
                            torch.save(self.Enc_eeg.state_dict(), model_path + sub_model_idx + 'Enc_eeg_cls.pth')
                            torch.save(self.Proj_imgf.state_dict(), model_path + sub_model_idx + 'Proj_imgf_cls.pth')

                print('Epoch:', e,
                      '  loss train: %.4f' % loss.detach().cpu().numpy(),
                      '  loss val: %.4f' % vloss.detach().cpu().numpy(),
                      )
                self.log_write.write('Epoch %d: loss train: %.4f, loss val: %.4f\n'%(e, loss.detach().cpu().numpy(), vloss.detach().cpu().numpy()))


        # * test part
        total = 0
        top1 = 0
        top3 = 0
        top5 = 0

        self.Enc_eeg.load_state_dict(torch.load(model_path + sub_model_idx + 'Enc_eeg_cls.pth'), strict=False)
        self.Proj_imgf.load_state_dict(torch.load(model_path + sub_model_idx + 'Proj_imgf_cls.pth'), strict=False)

        self.Enc_eeg.eval()
        self.Proj_imgf.eval()

        with torch.no_grad():
            for i, (teeg, tlabel) in enumerate(self.test_dataloader):
                teeg = Variable(teeg.to(self.args.device).type(self.Tensor))
                tlabel = Variable(tlabel.to(self.args.device).type(self.LongTensor))
                test_image_edge = Variable(test_image_edge.to(self.args.device).type(self.Tensor))
                test_image_depth = Variable(test_image_depth.to(self.args.device).type(self.Tensor))
                test_image_scene = Variable(test_image_scene.to(self.args.device).type(self.Tensor))

                ye, yd, ys = self.Enc_eeg(teeg)
                tfea = torch.cat([ye, yd, ys], dim=1)
                ze, zd, zs = self.Proj_imgf(test_image_edge, test_image_depth, test_image_scene)
                all_center = torch.cat([ze, zd, zs], dim=1)

                tfea = tfea / tfea.norm(dim=1, keepdim=True)
                all_center = all_center / all_center.norm(dim=1, keepdim=True)
                similarity = (100.0 * tfea @ all_center.t()).softmax(dim=-1)  # no use 100?
                _, indices = similarity.topk(5)

                tt_label = tlabel.view(-1, 1)
                total += tlabel.size(0)
                top1 += (tt_label == indices[:, :1]).sum().item()
                top3 += (tt_label == indices[:, :3]).sum().item()
                top5 += (tt_label == indices).sum().item()

            
            top1_acc = float(top1) / float(total)
            top3_acc = float(top3) / float(total)
            top5_acc = float(top5) / float(total)
        
        print('The test Top1-%.6f, Top3-%.6f, Top5-%.6f' % (top1_acc, top3_acc, top5_acc))
        self.log_write.write('The best epoch is: %d\n' % best_epoch)
        self.log_write.write('The test Top1-%.6f, Top3-%.6f, Top5-%.6f\n' % (top1_acc, top3_acc, top5_acc))
        
        return top1_acc, top3_acc, top5_acc
        # writer.close()


def main():

    print('seed is ' + str(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(result_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    num_sub = args.num_sub
    cal_num = 0
    aver = []
    aver3 = []
    aver5 = []

    for i in range(num_sub):
        cal_num += 1
        starttime = datetime.datetime.now()

        print('Subject %d' % (i+1))
        ie = IE(args, i + 1)

        Acc, Acc3, Acc5 = ie.train()
        print('THE BEST ACCURACY IS ' + str(Acc))

        endtime = datetime.datetime.now()
        print('subject %d duration: '%(i+1) + str(endtime - starttime))

        aver.append(Acc)
        aver3.append(Acc3)
        aver5.append(Acc5)

    aver.append(np.mean(aver))
    aver3.append(np.mean(aver3))
    aver5.append(np.mean(aver5))

    column = np.arange(1, cal_num+1).tolist()
    column.append('ave')
    pd_all = pd.DataFrame(columns=column, data=[aver, aver3, aver5])
    pd_all.to_csv(result_path + 'result.csv')

if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))