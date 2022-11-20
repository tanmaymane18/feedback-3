import unittest
from poolers import MeanPooling, MaxPooling, ClsPooler
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig

model_name = "microsoft/deberta-v3-base"
max_len = 1429
hidden_size = 768
hidden_dropout_prob = 0

class TestPoolers(unittest.TestCase):
    def setUp(self):

        config = AutoConfig.from_pretrained(model_name)
        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "add_pooling_layer": False
            }
        )

        base_model = AutoModel.from_pretrained(model_name, config=config)
        base_tok = AutoTokenizer.from_pretrained(model_name)

        toks = base_tok([
            "This is dummy text 1 ", 
            "This is dummy text 2"
            ], return_tensors="pt",
               max_length=max_len,
               padding=True)
        
        self.bs = 2
        self.last_n_cls = 2

        self.mean_pooler = MeanPooling()
        self.max_pooler = MaxPooling()
        self.cls_pooler = ClsPooler(hidden_size=hidden_size, last_n_cls=self.last_n_cls)
        self.embeddings = base_model(**toks)
        self.attention_mask = toks["attention_mask"]

        self.hidden_dim = self.embeddings.last_hidden_state.shape[2]

    def test_mean_pooler(self):
        mean_output = self.mean_pooler(self.embeddings, self.attention_mask)
        output_shape = mean_output.detach().numpy().shape
        self.assertEqual(output_shape, (self.bs, self.hidden_dim))
    
    def test_max_pooler(self):
        max_output = self.max_pooler(self.embeddings, self.attention_mask)
        output_shape = max_output.detach().numpy().shape
        self.assertEqual(output_shape, (self.bs, self.hidden_dim))
    
    def test_cls_pooler(self):
        cls_output = self.cls_pooler(self.embeddings)
        output_shape = cls_output.detach().numpy().shape
        self.assertEqual(output_shape, (self.bs, self.hidden_dim))

if __name__ == "__main__":
    unittest.main()
