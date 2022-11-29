import torch
import torch.nn as nn
from fastai.layers import TimeDistributed, Swish


class MeanPooling(nn.Module):
    def __init__(self, head=False):
        super(MeanPooling, self).__init__()
        self.head = head
        
    def forward(self, embeddings, attention_mask):
        if self.head:
            last_hidden_state = embeddings
            attention_mask = attention_mask
        else:
            last_hidden_state = embeddings.last_hidden_state
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class MaxPooling(nn.Module):
    def __init__(self, head=False):
        super(MaxPooling, self).__init__()
        self.head = head
        
    def forward(self, embeddings, attention_mask):
        if self.head:
            last_hidden_state = embeddings
            attention_mask = attention_mask
        else:
            last_hidden_state = embeddings.last_hidden_state
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        last_hidden_state[input_mask_expanded == 0] = -1e9
        max_embeddings = torch.max(last_hidden_state, 1)[0]
        return max_embeddings

class ClsPooler(nn.Module):
    def __init__(self, hidden_size, last_n_cls=4, weighted=False, drop_p=0):
        super(ClsPooler, self).__init__()
        self.last_n_cls = last_n_cls
        self.weighted = weighted
        self.hidden_size = hidden_size
        self.cls_pool_fc = LinLnDrop(hidden_size*self.last_n_cls, hidden_size, bias=False)
    
    def forward(self, embeddings, attention_mask=None):
        hidden_states = embeddings.hidden_states

        last_n_concat= torch.concat([
            hidden_states[-i][:,0,:] for i in range(1, self.last_n_cls+1)
        ], dim=-1)

        last_n_concat = self.cls_pool_fc(last_n_concat).unsqueeze(1)
        return last_n_concat

class MeanMaxPooler(nn.Module):
    def __init__(self, hidden_size=768):
        super(MeanMaxPooler, self).__init__()
        self.mean_pooler = MeanPooling(head=True)
        self.max_pooler = MaxPooling(head=True)
        self.tdl = TimeDistributed(module=nn.Linear(hidden_size, hidden_size, bias=False), low_mem=True, tdim=1)
        self.cls_fc = nn.Linear(hidden_size, hidden_size, bias=False)

        nn.init.kaiming_normal(self.cls_fc.weight)

    def forward(self, embeddings, attention_mask):
        last_hidden_state = embeddings.last_hidden_state
        tdl_output = self.tdl(last_hidden_state[:, 1:, :])
        mean_pooled = self.mean_pooler(tdl_output, attention_mask[:, 1:])
        max_pooled = self.max_pooler(tdl_output, attention_mask[:, 1:])
        cls_token = self.cls_fc(last_hidden_state[:, 0, :])

        pooled_embeds = torch.concat([
            cls_token, mean_pooled, max_pooled
        ], dim = -1)

        return pooled_embeds
class Conv1DLayer(nn.Sequential):
    def __init__(self, in_chn, out_chn, kernel_size=1, stride=2, bn=True, p=0):
        layers = []

        if bn:
            b_norm = nn.BatchNorm1d(in_chn)
            layers.append(b_norm)
        
        if p>0:
            drop_layer = nn.Dropout(p=p)
            layers.append(drop_layer)

        conv_1d = nn.Conv1d(
            in_channels=in_chn, 
            out_channels=out_chn,
            kernel_size=kernel_size,
            stride=stride
        )

        layers.append(conv_1d)
        super(Conv1DLayer, self).__init__(*layers)

class Conv1DBlock(nn.Sequential):
    def __init__(self, in_chn, out_chn, p=0):
        layers = [
            Conv1DLayer(in_chn=in_chn, out_chn=in_chn, p=p),
            Conv1DLayer(in_chn=in_chn, out_chn=out_chn, p=p),
            Swish()
        ]
    
        super(Conv1DBlock, self).__init__(*layers)

class ResNet1D(nn.Module):
    def __init__(self, in_chn, out_chn, p=0):
        super(ResNet1D, self).__init__()
        self.conv_1d_block = Conv1DBlock(in_chn=in_chn,
                                        out_chn=out_chn, p=p)
    
    def forward(self, x):
        conv_x = self.conv_1d_block(x)
        return conv_x + x

class ResNet1DBlock(nn.Sequential):
    def __init__(self, in_chn, out_chn, p=0):
        res_net_1d = ResNet1D(in_chn=in_chn, out_chn=in_chn, p=p)
        conv_block = Conv1DBlock(in_chn=in_chn, out_chn=out_chn)
        layers = [res_net_1d, conv_block]

        super(ResNet1DBlock, self).__init__(*layers)

class LinLnDrop(nn.Sequential):
    def __init__(self,in_dim, out_dim, act=None, norm_first=True, p=0, bias=True):
        layers = []
        if p > 0:
            layers.append(nn.Dropout(p))
        lin_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.kaiming_normal(lin_layer.weight)
        layers.append(lin_layer)
        if norm_first:
            layers.insert(0, nn.LayerNorm(in_dim))
        else:
            layers.append(nn.LayerNorm(out_dim))
        if act != None:
            layers.append(act)
        super(LinLnDrop, self).__init__(*layers)

class ResNetLinLnDrop(nn.Module):
    def __init__(self, in_size=768, p=0.2):
        super(ResNetLinLnDrop, self).__init__()        
        self.linear_block = nn.Sequential(*[
            LinLnDrop(in_dim=in_size, out_dim=in_size//2, act=Swish(), norm_first=True, p=p, bias=False),
            LinLnDrop(in_dim=in_size//2, out_dim=in_size, act=Swish(), norm_first=True, p=p, bias=False)
        ])
    
    def forward(self, x):
        x_out = self.linear_block(x)
        return x_out + x

class LinResBlock(nn.Sequential):
    def __init__(self, in_size=768, out_size=768, p=0.2, act=Swish()):
        layers = [
            ResNetLinLnDrop(in_size=in_size, p=p),
            LinLnDrop(in_dim=in_size, out_dim=out_size, act=act, norm_first=True, p=0, bias=False)
        ]
        
        super(LinResBlock, self).__init__(*layers)

if __name__ == "__main__":
    print("poolers")