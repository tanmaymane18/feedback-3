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

class cls_pooler(nn.Module):
    def __init__(self, last_n_cls=4):
        self.last_n_cls = last_n_cls
    
    def forward(self, embeddings):
        hidden_states = embeddings.hidden_states
        return torch.concat([
            hidden_states[-i][:,0,:] for i in range(1, self.last_n_cls+1)
        ])




if __name__ == "__main__":
    print("poolers")