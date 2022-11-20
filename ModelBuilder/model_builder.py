from .poolers import *
from fastai.layers import TimeDistributed, LinBnDrop
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

class ModelBuilder:
    def __init__(self):
        pass

    def build_body(self):
        pass

    def build_pooler(self):
        pass

    def build_head(self):
        pass

    def get_model(self):
        pass
