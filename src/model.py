import logging
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

class Model(nn.Module):

    def __init__(self, num_classes, input_size, num_layers, alpha_l1_loss, alpha_group_sparsity_loss,
                 alpha_pseudo_gt_loss, weighted_psgt_loss, pseudo_gt_loss_dim):
        '''
        Args:
            num_classes (int): The number of labels to classify
            input_size (int): The size of each feature
            alpha_l1_loss (float): constant factor to weight the l1 loss
            alpha_group_sparsity_loss (float): constant factor to weight the group sparsity loss
            alpha_pseudo_gt_loss (float): constant factor to weight the pseudo gt loss
        '''
        super().__init__()
        logging.info(f'Model: num_classes {num_classes} input_size {input_size} num_layers {num_layers} '
                     f'alpha_l1_loss {alpha_l1_loss} alpha_group_sparsity_loss {alpha_group_sparsity_loss} '
                     f'alpha_pseudo_gt_loss {alpha_pseudo_gt_loss} weighted_psgt_loss {weighted_psgt_loss} '
                     f'pseudo_gt_loss_dim {pseudo_gt_loss_dim}' )
        self.num_classes = num_classes
        self.input_size = input_size
        self.pseudo_gt_loss_dim = pseudo_gt_loss_dim
        self.alpha_l1_loss = alpha_l1_loss
        self.alpha_group_sparsity_loss = alpha_group_sparsity_loss
        self.alpha_pseudo_gt_loss = alpha_pseudo_gt_loss
        self.weighted_psgt_loss = weighted_psgt_loss
        self.num_layers = num_layers
        self.fc_cas = Model._create_sequential_linear_relu_layers(self.num_layers, self.input_size, self.num_classes)
        self.fc_attention = Model._create_sequential_linear_relu_layers(self.num_layers, self.input_size, self.pseudo_gt_loss_dim)

    @staticmethod
    def _create_sequential_linear_relu_layers(num_layers, input_size, output_size):
        layers = OrderedDict()
        this_input_size = input_size
        for i in range(num_layers-1):
            this_output_size = max(int(this_input_size/2), output_size)
            layers[f'linear_{i}'] = nn.Linear(this_input_size, this_output_size)
            layers[f'relu_{i}'] = nn.ReLU()
            this_input_size = this_output_size

        layers[f'linear_{num_layers-1}'] = nn.Linear(this_input_size, output_size)
        return nn.Sequential(layers)

    def forward(self, features):
        cas = self.fc_cas(features)
        attention = self.fc_attention(features)
        return cas, attention

    @staticmethod
    def _l1_loss(x): return torch.mean(torch.abs(x))

    @staticmethod
    def _group_sparsity_loss(x): return torch.mean(torch.abs(x[:,1:] - x[:,:-1]))

    def loss(self, cas, attention, targets, device):
        loss_results = {'loss': torch.tensor(0.0).to(device),
                        'video_label_loss': torch.tensor(0.0).to(device),
                        'l1_loss': torch.tensor(0.0).to(device),
                        'group_sparsity_loss': torch.tensor(0.0).to(device),
                        'pseudo_gt_loss': torch.tensor(0.0).to(device)}

        for i, video_name in enumerate(targets['video-names']):
            start, end = targets['video-name-to-slice'][video_name]
            this_cas = cas[start:end].unsqueeze(0)
            this_attention = attention[start:end].unsqueeze(0)
            this_labels = targets['labels'][i:i+1].unsqueeze(0)
            this_pseudo_gt = targets['pseudo-gt'][start:end].unsqueeze(0)

            this_loss_results = self._loss_for_one_video(cas=this_cas, attention=this_attention, labels=this_labels, 
                                                        pseudo_gt=this_pseudo_gt, 
                                                        fg_class_loss_weight=targets['fg_class_loss_weight'],
                                                        device=device)
            for k, v in this_loss_results.items():
                loss_results[k] += v

        batch_size = len(targets['video-names'])
        for k, v in loss_results.items():
            loss_results[k] /= batch_size

        return loss_results

    def _loss_for_one_video(self, cas, attention, labels, pseudo_gt, fg_class_loss_weight, device):
        '''
            cas: 1 x T x num_classes
            attention" 1 x T x {1,2}
            labels: 1 x 1
            pseudo_gt: 1 x T
            bg_cas: 1 x T X num_classes+1
        '''

        softmax_attention_across_time, softmax_attention_across_class = smooth_attention_function(attention)
        softmax_cas = F.log_softmax(cas, dim=-1)
        attention_cas = torch.sum(softmax_attention_across_time * softmax_cas, dim=-2)

        video_label_loss = F.nll_loss(attention_cas, torch.squeeze(labels, dim=-1).to(device))
        l1_loss = Model._l1_loss(attention)
        group_sparsity_loss = Model._group_sparsity_loss(softmax_attention_across_time)

        idx = pseudo_gt >= 0

        if idx.byte().any():
            if self.weighted_psgt_loss:
                if self.pseudo_gt_loss_dim == 2:
                    pseudo_gt_loss = F.cross_entropy(input=attention[idx], target=pseudo_gt[idx].to(device),
                                                     weight=torch.tensor([1.0 - fg_class_loss_weight, fg_class_loss_weight]).to(device))
                elif self.pseudo_gt_loss_dim == 1:
                    pseudo_gt_loss = F.binary_cross_entropy_with_logits(input=attention[idx].view(-1),
                                                    target=pseudo_gt[idx].type(torch.cuda.FloatTensor).to(device),
                                                    pos_weight=torch.tensor(np.sqrt(fg_class_loss_weight)).to(device))
                else:
                    raise ValueError('Got a value of {} dimensions. Only valid dimensions are 1 and 2'.format(self.attention_dim))
            else:
                if self.pseudo_gt_loss_dim == 2:
                    pseudo_gt_loss = F.cross_entropy(input=attention[idx], target=pseudo_gt[idx].to(device))
                elif self.pseudo_gt_loss_dim == 1:
                    pseudo_gt_loss = F.binary_cross_entropy_with_logits(input=attention[idx].view(-1),
                                                    target=pseudo_gt[idx].type(torch.cuda.FloatTensor).to(device))
        else:
            pseudo_gt_loss = torch.tensor(0.0).to(device)

        loss = video_label_loss + self.alpha_l1_loss*l1_loss + self.alpha_group_sparsity_loss*group_sparsity_loss \
               + self.alpha_pseudo_gt_loss*pseudo_gt_loss

        loss_results = {'loss': loss,
                        'video_label_loss': video_label_loss,
                        'l1_loss': l1_loss,
                        'group_sparsity_loss': group_sparsity_loss,
                        'pseudo_gt_loss': pseudo_gt_loss}
        return loss_results

def smooth_attention_function(input_vector):
    if input_vector.shape[-1] == 2:
        out_attention_class = F.softmax(input_vector, dim=-1)
        out_attention_time = F.softmax(out_attention_class, dim=-2)[..., 1:2]
        out_attention_class = out_attention_class[...,1:]
    elif input_vector.shape[-1] == 1:
        out_attention_time = F.softmax(input_vector, dim=-2)
        out_attention_class = torch.sigmoid(input_vector)
    else:
        raise ValueError('Invalid vector shape')
    return out_attention_time, out_attention_class