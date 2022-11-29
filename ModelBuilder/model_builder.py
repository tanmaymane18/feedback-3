import torch
import torch.nn as nn
from fastai.layers import TimeDistributed, LinBnDrop, Swish, Mish, sigmoid_range
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

from .poolers import MeanPooling, MaxPooling, ClsPooler, LinLnDrop, LinResBlock

class FeedbackModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = None
        self.pooler = None
        self.head = None
    
    def forward(self, input_ids, attention_mask):
        body_output = self.body(input_ids, attention_mask)
        pooler_output = self.pooler(body_output, attention_mask)
        output = self.head(pooler_output)

        return SequenceClassifierOutput(logits=output)

class FeedbackPooler(nn.Module):
    def __init__(self, poolers=[]):
        super(FeedbackPooler, self).__init__()
        self.is_poolers = False
        if len(poolers):
            self.is_poolers = True
            self.poolers = nn.ModuleList([
                pooler() for pooler in poolers
            ])
    
    def forward(self, embeddings, attention_mask):
        combine_feature = None
        combine_feature = embeddings.last_hidden_state[:, 0, :]
        combine_feature = combine_feature.unsqueeze(1)
        if self.is_poolers:
            for layer in self.poolers:
                layer_output = layer(embeddings, attention_mask)
                if len(layer_output.shape)<3:
                    layer_output = layer_output.unsqueeze(1)
                if combine_feature != None:
                    combine_feature = torch.concat([combine_feature, layer_output], dim=-1)
                else:
                    combine_feature = layer_output
        
        return combine_feature

class FeedbackHead(nn.Module):  
    def __init__(self, dims, ps, n_blocks=3):
        super(FeedbackHead, self).__init__()
        # acts = [Swish()] * (len(dims) - 2) + [None]
        # layers = [LinLnDrop(i, o, p=p, act=a) for i,o,p,a in zip(dims[:-1], dims[1:], ps, acts)] + [LinBnDrop(dims[-1], 6, bn=False)]
        # self.layers = nn.Sequential(*layers)
        # sizes = [768*4, 768*2, 768, 768]
        # self.layers = nn.Sequential(*[
        #     LinResBlock(in_size=sizes[i], out_size=sizes[i+1]) for i in range(n_blocks)
        # ])
        self.drop_1 = nn.Dropout(0.1)
        self.drop_2 = nn.Dropout(0.2)
        self.drop_3 = nn.Dropout(0.3)
        self.drop_4 = nn.Dropout(0.4)
        self.drop_5 = nn.Dropout(0.5)
        self.output_layer = LinLnDrop(768, 6, bias=True)
    
    def forward(self, x):
        x_out = x.squeeze(1)
        drop_1 = self.output_layer(self.drop_1(x_out))
        drop_2 = self.output_layer(self.drop_2(x_out))
        drop_3 = self.output_layer(self.drop_3(x_out))
        drop_4 = self.output_layer(self.drop_4(x_out))
        drop_5 = self.output_layer(self.drop_5(x_out))
        x_out = (drop_1 + drop_2 + drop_3 + drop_4 + drop_5)/5
        x_out = sigmoid_range((x_out), 0.5,5.5)
        return x_out

class ModelBuilder:
    def __init__(self, model_name, dims, ps, poolers=None, num_labels=6, hidden_dropout_prob=0):
        self.model_name = model_name
        self.num_labels = num_labels
        self.dims = dims
        self.ps = ps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.poolers = poolers
        self.model = FeedbackModel()

    def build_body(self):
        config = AutoConfig.from_pretrained(self.model_name)
        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": self.hidden_dropout_prob,
                "add_pooling_layer": False,
                "num_labels": self.num_labels,
            }
        )
        base_model = AutoModel.from_pretrained(self.model_name, config=config)
        base_model.gradient_checkpointing_enable()

        return base_model

    def build_pooler(self):
        return FeedbackPooler(self.poolers)

    def build_head(self):
        head = FeedbackHead(dims=self.dims, ps=self.ps)

        return head

    def get_model(self):
        self.model.body = self.build_body()
        self.model.pooler = self.build_pooler()
        self.model.head = self.build_head()

        return self.model


if __name__ == "__main__":
    print("model builder")