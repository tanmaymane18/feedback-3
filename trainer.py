import os
import numpy as np
import pandas as pd
from functools import partial
from fastai.text.all import *
from Managers.managers import ExpManager
from ModelBuilder.poolers import ClsPooler, MeanPooling, MaxPooling
from transformers import logging
logging.set_verbosity_error()
import warnings

warnings.filterwarnings("ignore")

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

def params(m):
    return [p for p in m.parameters()]

def transSplitter(model):
    groups = L(model.body.base_model.embeddings.children())
    for layer in list(model.body.base_model.encoder.layer.children()):
        groups = groups + L(layer)
    groups = groups + L(m for m in list(model.pooler.children()) if params(m)) + L(m for m in list(model.head.children()) if params(m))
    return groups.map(params)

def MCRMSE_metric(outputs, targets):
    colwise_mse = torch.mean(torch.square(targets - outputs), dim=0)
    loss = torch.mean(torch.sqrt(colwise_mse), dim=0)
    return loss

class RMSE:
    def __init__(self, reduction='mean', eps=1e-9):
        store_attr()
        self.mse_loss = MSELossFlat(reduction='none')
    
    def __call__(self, pred, targ):
        mse_loss = self.mse_loss(pred, targ)
        rmse = torch.sqrt(mse_loss+self.eps)
        if self.reduction == "mean":
            return rmse.mean()
        if self.reduction == "sum":
            return rmse.sum()
        return rmse

MCRMSE = AccumMetric(MCRMSE_metric)

def set_seed(x=42):
    random.seed(x)
    np.random.seed(x)
    torch.manual_seed(x)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(x)

hidden_size = 768

config = dict(
    model_name="microsoft/deberta-v3-base",
    max_len=1429,
    num_folds=5,
    num_labels=6,
    hidden_dropout_prob=0,
    seed=42,
    bs=8,
    val_bs=16,
    model_splitter=transSplitter,
    metric=MCRMSE,
    opt_func=Adam,
    loss_func=RMSE,
    fine_tune=True,
    n_epochs=2,
    lr=3e-2,
    pct_start=0.25,
    wd=0.01,
    model_dir="test_run_models",
    poolers=[
        partial(ClsPooler, hidden_size=hidden_size, 
                        last_n_cls=4, drop_p=0),
        MeanPooling,
        MaxPooling
    ],
    dims=[
        (hidden_size*3),
        (hidden_size*3)//2,
        (hidden_size*3)//4,
        6
    ],
    ps = [
        0.2,
        0.3,
        0.4,
        0
    ],
    freeze_to=15,
    fit_type="flat_cos"
)

if __name__ == "__main__":

    set_seed()

    df = pd.read_csv("data\\train_folds.csv")

    exp_manager = ExpManager(
        data=df,
        **config
    )

    exp_manager.model_setup()
    exp_manager.fold_trainings()
    exp_manager.oof_pred_eval()