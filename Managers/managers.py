import gc
from copy import deepcopy
from fastai.text.all import *
from fasthugs.data import TransformersTextBlock
from fasthugs.learner import TransLearner
from ModelBuilder.model_builder import ModelBuilder

class ExpManager:

    def __init__(self, model_name, data, max_len, num_folds=5, **kwargs):
        self.data = data
        self.num_folds = num_folds
        self.model_name = model_name
        self.max_len = max_len
        self.oof_preds = []
        self.oof_targs = []
        self.kwargs = kwargs

    def model_setup(self):
        self.model_builder = ModelBuilder(
            model_name=self.model_name, 
            dims=self.kwargs["dims"], 
            ps=self.kwargs["ps"], 
            poolers=self.kwargs["poolers"], 
            num_labels=self.kwargs["num_labels"], 
            hidden_dropout_prob=self.kwargs["hidden_dropout_prob"]
        )

    def fold_trainings(self):
        torch.cuda.empty_cache()
        gc.collect()
        for fold in range(self.num_folds):
            print("\n\n")
            print("#"*30)
            print("#")
            print(f"#\tFold: {fold+1}")
            print("#")
            print("#"*30)
            print("\n\n")

            val_idx = self.data[self.data["kfold"]==fold].index.to_list()

            model = self.model_builder.get_model()

            dblock = DataBlock(
                                blocks = [TransformersTextBlock(pretrained_model_name=self.model_name, 
                                                                max_length=self.max_len,
                                                            padding='max_length'),
                                        RegressionBlock()],
                                get_x=ColReader(cols=["full_text"]),
                                get_y=ColReader(cols=["cohesion", "syntax", 
                                                      "vocabulary", "phraseology", 
                                                      "grammar", "conventions"]),
                                splitter=IndexSplitter(val_idx)
                            )
            
            dls = dblock.dataloaders(
                                        self.data, 
                                        bs = self.kwargs["bs"],
                                        val_bs = self.kwargs["val_bs"]
                                    )
            dls.rng.seed(self.kwargs["seed"])

            learn = TransLearner(
                dls, 
                model, 
                splitter=self.kwargs["model_splitter"], 
                metrics=[self.kwargs["metric"]], 
                opt_func=self.kwargs["opt_func"], 
                loss_func=self.kwargs["loss_func"]()
            )
            # learn = learn.to_fp16()

            if self.kwargs["fine_tune"]:
                learn.freeze_to(self.kwargs["freeze_to"])
                if self.kwargs["fit_type"] == "flat_cos":
                    learn.fit_flat_cos(
                        self.kwargs["n_epochs"],
                        lr=self.kwargs["lr"],
                        pct_start=self.kwargs["pct_start"],
                        wd=self.kwargs["wd"],
                        cbs=[GradientClip]
                    )
                elif self.kwargs["fit_type"] == "one_cycle":
                    lr_factor = 1
                    for _ in range(self.kwargs["num_fits"]):
                        learn.fit_one_cycle(
                            self.kwargs["n_epochs"],
                            lr_max=self.kwargs["lr"]*lr_factor,
                            pct_start=self.kwargs["pct_start"],
                            wd=self.kwargs["wd"],
                            cbs=[GradientClip],
                            moms=self.kwargs["moms"]
                        )

                        # sched = {"lr": combined_cos(0.20, self.kwargs["lr"]/100, self.kwargs["lr"], 0)}

                        # learn.fit(
                        #     self.kwargs["n_epochs"],
                        #     wd=self.kwargs["wd"],
                        #     cbs=[ParamScheduler(sched), GradientClip]
                        # )

                        lr_factor = lr_factor*0.9
            
            if "full_training" in self.kwargs:
                if self.kwargs["full_training"]:
                    learn.unfreeze()
                    print("unfreezed all layers")
                    learn.fit_one_cycle(
                            self.kwargs["n_epochs"],
                            lr_max=slice(self.kwargs["lr_min"], self.kwargs["lr_max"]),
                            pct_start=self.kwargs["pct_start"],
                            wd=self.kwargs["wd"]
                        )
            
            val_preds, val_targs = learn.get_preds(ds_idx=1)

            val_metric = self.kwargs["metric"](val_preds, val_targs)

            print(f"#\tval_metric: {val_metric}\t#")

            self.oof_preds.append(val_preds)
            self.oof_targs.append(val_targs)

            learn.model_dir = self.kwargs["model_dir"]
            # learn.export(f"fb3_fold_{fold+1}.pkl", with_opt=False)
            learn.export(f"fb3_fold_{fold+1}.pkl")

            del learn, dls, dblock, model
            torch.cuda.empty_cache()
            gc.collect()

    def oof_pred_eval(self):
        oof_preds = torch.concat(self.oof_preds)
        oof_targs = torch.concat(self.oof_targs)

        oof_metric = self.kwargs["metric"](oof_preds, oof_targs)

        print(f"\n#\tOOF CV: {oof_metric.item()}\t#")

        return oof_metric


if __name__ == "__main__":
    print("train manager")