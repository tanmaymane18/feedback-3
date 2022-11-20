from poolers import *
from fastai.layers import TimeDistributed, LinBnDrop, Swish, Mish
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

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
    def __init__(self, poolers):
        super(FeedbackPooler, self).__init__()
        self.poolers = nn.ModuleList([
            pooler() for pooler in poolers
        ])
    
    def forward(self, embeddings, attention_mask):
        combine_feature = None
        for layer in self.poolers:
            layer_output = layer(embeddings, attention_mask)
            layer_output = layer_output.unsqueeze(1)
            if combine_feature != None:
                combine_feature = torch.concat([combine_feature, layer_output], dim=-1)
            else:
                combine_feature = layer_output
        
        return combine_feature

class FeedbackHead(nn.Module):
    def __init__(self, dims, ps):
        super(FeedbackHead, self).__init__()
        acts = [nn.ReLU(inplace=True)] * (len(dims) - 2) + [None]
        layers = [LinBnDrop(i, o, p=p, act=a) for i,o,p,a in zip(dims[:-1], dims[1:], ps, acts)]
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.squeeze(1)
        x = self.layers(x)
        return SequenceClassifierOutput(logits=x)

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