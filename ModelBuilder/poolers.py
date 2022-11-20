import torch
import torch.nn as nn


class MeanPooling(nn.Module):
    def __init__(self, head=False):
        super(MeanPooling, self).__init__()
        self.head = head
        
    def forward(self, embeddings, attention_mask):
        last_hidden_state = embeddings.last_hidden_state
        if self.head:
            last_hidden_state = embeddings.last_hidden_state[:, 1:, :]
            attention_mask = attention_mask[:, 1:, :]
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
        last_hidden_state = embeddings.last_hidden_state
        if self.head:
            last_hidden_state = embeddings.last_hidden_state[:, 1:, :]
            attention_mask = attention_mask[:, 1:, :]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        last_hidden_state[input_mask_expanded == 0] = -1e9
        max_embeddings = torch.max(last_hidden_state, 1)[0]
        return max_embeddings

class ClsPooler(nn.Module):
    def __init__(self, hidden_size, last_n_cls=4, drop_p=0):
        super(ClsPooler, self).__init__()
        self.last_n_cls = last_n_cls
        self.cls_pool_fc = nn.Linear(hidden_size*last_n_cls, hidden_size, bias=False)
    
    def forward(self, embeddings, attention_mask=None):
        hidden_states = embeddings.hidden_states
        last_n_concat= torch.concat([
            hidden_states[-i][:,0,:] for i in range(1, self.last_n_cls+1)
        ], -1)
        last_n_concat = self.cls_pool_fc(last_n_concat)
        return last_n_concat


if __name__ == "__main__":
    print("poolers")