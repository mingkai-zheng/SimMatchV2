# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, base_encoder, num_classes, dim=128, args=None):
        super(ResNet, self).__init__()
        self.backbone = base_encoder()
        assert not hasattr(self.backbone, 'fc'), "fc should not in backbone"

        self.fc = nn.Linear(self.backbone.out_channels, num_classes)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.out_channels, self.backbone.out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.backbone.out_channels, dim),
        )
        
        if args.norm_feat:
            self.norm = nn.LayerNorm(self.backbone.out_channels)
        else:
            self.norm = nn.Identity()
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.norm(x)
        logits = self.fc(x)
        embedding = self.head(x)
        embedding = F.normalize(embedding)
        return logits,  embedding


class SimMatchV2(nn.Module):
    def __init__(self, base_encoder, num_classes=1000, eman=False, momentum=0.999, dim=128, norm=None, label_bank=None, K=256*320, args=None):
        super(SimMatchV2, self).__init__()
        self.eman = eman
        self.momentum = momentum
        self.num_classes = num_classes

        self.dim = dim
        self.main = ResNet(base_encoder, num_classes, dim=dim, args=args)
        self.ema  = ResNet(base_encoder, num_classes, dim=dim, args=args)


        for param_main, param_ema in zip(self.main.parameters(), self.ema.parameters()):
            param_ema.data.copy_(param_main.data)  # initialize
            param_ema.requires_grad = False  # not update by gradient

        self.K = K

        # memory bank to store the embeding for labeled data
        self.register_buffer("l_bank", torch.randn(label_bank.size(0), dim))
        self.l_bank = nn.functional.normalize(self.l_bank)

        # memory bank to store the onehot label for labeled data
        self.register_buffer("l_labels", torch.zeros(label_bank.size(0), num_classes).scatter(1, label_bank.unsqueeze(1), 1))
    
        # unlabaled memory bank pointer
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

        # memory bank to store the embeding for unlabeled data
        self.register_buffer("u_bank", torch.randn(self.K, dim))
        self.u_bank = nn.functional.normalize(self.u_bank)

        # memory bank to store the model's prediction for unlabeled data
        self.register_buffer("u_labels", torch.zeros(self.K, num_classes) / num_classes)
        
        # distrbutional alignment
        self.register_buffer("da", torch.ones([1,num_classes], dtype=torch.float) / num_classes)


    def momentum_update_ema(self, m):
        if self.eman:
            state_dict_main = self.main.state_dict()
            state_dict_ema = self.ema.state_dict()
            for (k_main, v_main), (k_ema, v_ema) in zip(state_dict_main.items(), state_dict_ema.items()):
                assert k_main == k_ema, "state_dict names are different!"
                assert v_main.shape == v_ema.shape, "state_dict shapes are different!"
                if 'num_batches_tracked' in k_ema:
                    v_ema.copy_(v_main)
                else:
                    v_ema.copy_(v_ema * m + (1. - m) * v_main)
        else:
            for param_q, param_k in zip(self.main.parameters(), self.ema.parameters()):
                param_k.data = param_k.data * m + param_q.data * (1. - m)
    

    @torch.no_grad()
    def distribution_alignment(self, probs, args):
        da_prob = probs / self.da
        da_prob = da_prob / da_prob.sum(dim=1, keepdim=True)
        batch_prob = torch.sum(probs, dim=0, keepdim=True)
        torch.distributed.all_reduce(batch_prob)
        batch_prob = batch_prob / (probs.size(0) * torch.distributed.get_world_size())
        self.da = self.da * args.da_m + batch_prob * (1 - args.da_m)
        return da_prob.detach()
    
    
    @torch.no_grad()
    def _update_unlabel_bank(self, k, prob):
        k = concat_all_gather(k)
        prob = concat_all_gather(prob)
        batch_size = k.size(0)
        ptr = int(self.ptr[0])
        assert self.K % batch_size == 0
        self.u_bank[ptr:ptr + batch_size] = k
        self.u_labels[ptr:ptr + batch_size] = prob
        self.ptr[0] = (ptr + batch_size) % self.K
    
    @torch.no_grad()
    def _update_label_bank(self, k, index):
        k     = concat_all_gather(k)
        index = concat_all_gather(index)
        self.l_bank[index] = k


    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]


    def forward(self, im_x, im_u_w=None, im_u_s=None, index=None, args=None):
        if im_u_w is None and im_u_s is None:
            logits, _ = self.main(im_x)
            return logits

        device = im_x.device

        l_bank   = self.l_bank.clone().detach()
        l_labels = self.l_labels.clone().detach()
        u_bank   = self.u_bank.clone().detach()
        u_labels = self.u_labels.clone().detach()

        batch_x = im_x.shape[0]
        batch_u = im_u_w.shape[0]

        num_strong = len(im_u_s)
        
        with torch.no_grad():
            im = torch.cat([im_x, im_u_w])
            if self.eman:
                logits_k, feat_k = self.ema(im)
            else:
                im, idx_unshuffle = self._batch_shuffle_ddp(im)
                logits_k, feat_k = self.ema(im)
                logits_k = self._batch_unshuffle_ddp(logits_k, idx_unshuffle)
                feat_k   = self._batch_unshuffle_ddp(feat_k , idx_unshuffle)

            feat_kx = feat_k[:batch_x]
            feat_ku = feat_k[batch_x:]
            prob_ku = F.softmax(logits_k[batch_x:], dim=1)
            if args.DA:
                prob_ku = self.distribution_alignment(prob_ku, args)
        
            simmatrix_k = feat_ku @ u_bank.T
            relation_ku = F.softmax(simmatrix_k / args.t, dim=-1)
        
            l_sim_index = torch.topk(feat_ku @ l_bank.T, k=args.topn, largest=True, sorted=False, dim=-1)[1].flatten()
            l_nearest_neighbor_feat = l_bank[l_sim_index].reshape(batch_u, args.topn, -1)        #[batch_u, topn  , feat]
            l_onehot_label = l_labels[l_sim_index].reshape(batch_u, args.topn, -1)               #[batch_u, topn  , num_classes]
            l_concat_feature = torch.cat([feat_ku.unsqueeze(1), l_nearest_neighbor_feat], dim=1) #[batch_u, topn+1, feat]
            l_concat_label   = torch.cat([prob_ku.unsqueeze(1), l_onehot_label], dim=1)          #[batch_u, topn+1, num_classes]
            
            masks = torch.eye(args.topn+1, device=device)
            A = l_concat_feature @ l_concat_feature.transpose(-2, -1) / args.t - masks * 1e9       #[batch_u, topn+1, topn+1]
            A = F.softmax(A, dim=-1)                                                               #[batch_u, topn+1, topn+1]
            A = (1 - args.alpha) * torch.inverse(masks - args.alpha * A)                           #[batch_u, topn+1, topn+1]
            pseudo_label = (A @ l_concat_label)[:, 0]                                                     #[batch_u, topn+1, num_class][:, 0]


        logits_q, feat_q = self.main(torch.cat([im_x, im_u_s[0]]))
        logits_qx, logits_qu = logits_q[:batch_x], logits_q[batch_x:]
        feat_qu = feat_q[batch_x:]
        logits_qu_list = [logits_qu]
        feat_qu_list = [feat_qu]
        for im_q in im_u_s[1:]:
            logits_qu, feat_qu = self.main(im_q)
            logits_qu_list.append(logits_qu)
            feat_qu_list.append(feat_qu)
        
        loss_ee = 0
        loss_ne = 0
        for idx in range(num_strong):
            relation_qu = F.softmax(feat_qu_list[idx] @ u_bank.T / args.t, dim=1)
            loss_ee += torch.sum(-relation_qu.log() * relation_ku.detach(), dim=1).mean()

            nn_qu = relation_qu @ u_labels
            loss_ne += torch.sum(-nn_qu.log() * prob_ku.detach(), dim=1).mean()
        
        loss_ee /= num_strong
        loss_ne /= num_strong
        
        if torch.isnan(loss_ee) or torch.isinf(loss_ee):
            loss_ee = torch.zeros(1, device=device)

        if torch.isnan(loss_ne) or torch.isinf(loss_ne):
            loss_ne = torch.zeros(1, device=device)
    
        self._update_label_bank(feat_kx, index)
        self._update_unlabel_bank(feat_ku, prob_ku)

        logits_qu = torch.cat(logits_qu_list)
        return logits_qx, logits_qu, loss_ee, loss_ne, prob_ku.detach().repeat([num_strong, 1]), pseudo_label.detach().repeat([num_strong, 1])

        


def get_simmatch_model(model):
    if isinstance(model, str):
        model = {
            "SimMatchV2": SimMatchV2,
        }[model]
    return model



# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)

    output = torch.cat(tensors_gather, dim=0)
    return output

