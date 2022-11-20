import torch
import torch.nn as nn
from fastai.layers import LinBnDrop, TimeDistributed
from .poolers import  MeanPooling, MaxPooling


class MeanMaxHead(nn.Module):
    def __init__(self, layer, hidden_size, num_labels, fc_p_out=0, *args):
        super(MeanMaxHead, self).__init__()
        self.mean_pooler = MeanPooling(head=True)
        self.max_pooler = MaxPooling(head=True)
        self.tdl = TimeDistributed(layer, tdim=1)
        self.cls_fc = LinBnDrop(n_in=hidden_size, n_out=hidden_size)
        self.fc = LinBnDrop(n_in = hidden_size*3-2, n_out=num_labels, p=fc_p_out)
    
    def forward(self, embeddings, attention_mask):
        last_hidden_state = embeddings.last_hidden_state
        cls_token = last_hidden_state[:, 0, :]
        cls_output = self.cls_fc(cls_token)
        tdl_output = self.tdl(last_hidden_state[:, 1:, :])
        mean_features = self.mean_pooler(tdl_output, attention_mask[:, 1:, :])
        max_features = self.max_pooler(tdl_output, attention_mask[:, 1:, :])
        concat_feats = torch.concat([cls_output, mean_features, max_features], dim = -1)
        output = self.fc(concat_feats)
        return output

if __name__ == "__main__":
    print("heads")