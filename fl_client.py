# import argparse
# import subprocess

# import flwr as fl
# import torch
# import os
# import pandas as pd
# from main import Experiment  # your existing Experiment class
# from model import HE2RNA
# from torch.nn.modules.conv import Conv1d
# torch.serialization.add_safe_globals([HE2RNA, Conv1d])

# class HE2RNAClient(fl.client.NumPyClient):
#     def __init__(self, args):
#         self.args = args

#         # 1⃣ Extract ResNet tile features locally
#         # subprocess.run(
#         #     [
#         #         "python",
#         #         "extract_tile_features_from_slides.py",
#         #         "--path_to_slides", args.path_to_slides,
#         #         "--tile_coordinates", args.tile_coordinates,
#         #         "--path_to_save_features", args.path_to_save_features,
#         #     ],
#         #     check=True,
#         # )

#         # 2⃣ (Optional) any transcriptome_data.py preprocessing step
#         # subprocess.run(["python", "transcriptome_data.py"], check=True)

#         # # 3⃣ Build supertiles locally
#         # subprocess.run(
#         #     [
#         #         "python",
#         #         "supertile_preprocessing.py",
#         #         "--path_to_transcriptome",           args.path_to_transcriptome,
#         #         "--path_to_save_processed_data",     args.path_to_save_processed_data,
#         #         "--n_tiles",                         str(args.n_tiles),
#         #     ],
#         #     check=True,
#         # )

#         # 4⃣ Initialize your Experiment object (reads config, but does _not_ train yet)
#         self.exp = Experiment(args.config)
#         self.model_path = os.path.join("all_genes", "model.pt")  
#         # → resolves to "all_genes/model.pt"
        
#     def _safe_load_model(self):
#         """
#         Try to torch.load(model.pt). On failure, delete model.pt
#         and regenerate a fresh checkpoint via single_run().
#         Returns the loaded model.
#         """
#         try:
#             # In PyTorch ≥2.6, default weights_only=True loads a state_dict.
#             # But we explicitly allow full-model unpickling if that's what you saved.
#             model = torch.load(self.model_path, weights_only=False)
#             return model
#         except Exception as e:
#             print(f"⚠️  Error loading '{self.model_path}': {e}")
#             print("→ Removing corrupt checkpoint and regenerating from scratch.")
#             try:
#                 os.remove(self.model_path)
#             except OSError:
#                 pass

#             # Run your single‐run to recreate model.pt
#             self.exp.single_run(logdir=self.args.logdir)

#             # Load again, this time it should succeed
#             model = torch.load(self.model_path, weights_only=False)
#             return model
        
#     def get_parameters(self,config):
#         # Return initial model weights as a list of NumPy arrays
#         # params = [
#         #     val.cpu().numpy()
#         #     for val in self.exp.single_run.__self__.model.state_dict().values()
#         # ]
#         # return params
#     #     """Return current global model weights from all_genes/model.pt."""
#     #     # If no model exists yet, create it by running one single_run
#         if not os.path.exists(self.model_path):
#             self.exp.single_run(logdir=self.args.logdir)
#         # model = torch.load(self.model_path,weights_only=False)
#         model = self._safe_load_model()

#         return [val.cpu().numpy() for val in model.state_dict().values()]


#     def fit(self, parameters, config):
#     #     # 1) Load round‑zero parameters into your model
#     #     state_dict = self.exp.single_run.__self__.model.state_dict()
#     #     for k, new_val in zip(state_dict.keys(), parameters):
#     #         state_dict[k] = torch.tensor(new_val)
#     #     self.exp.single_run.__self__.model.load_state_dict(state_dict)

#     #     # 2) Run a local training epoch (or full single_run)
#     #     self.exp.single_run(logdir=args.logdir)

#     #     # 3) Extract updated weights
#     #     new_params = [
#     #         val.cpu().numpy()
#     #         for val in self.exp.single_run.__self__.model.state_dict().values()
#     #     ]

#     #     # 4) Return them plus number of training examples
#     #     num_examples = len(self.exp._build_dataset())
#     #     return new_params, num_examples, {}
#         """Load global, train locally via main.py CLI, return updated weights."""
#         # 1) Overwrite local model with global weights
#         if os.path.exists(self.model_path):
#             model = torch.load(self.model_path,weights_only=False)
#             state_dict = model.state_dict()
#             for k, new_val in zip(state_dict.keys(), parameters):
#                 state_dict[k] = torch.tensor(new_val)
#             model.load_state_dict(state_dict)
#             torch.save(model, self.model_path)

#         # 2) Run your two CLI commands
#         # if self.args.run == "single_run":
#         #     subprocess.run([
#         #         "python", "main.py",
#         #         "--config", self.args.config,
#         #         "--run",    "single_run",
#         #         "--logdir", self.args.logdir,
#         #     ], check=True)
#         # else:
#         #     subprocess.run([
#         #         "python", "main.py",
#         #         "--config",   self.args.config,
#         #         "--run",      "cross_validation",
#         #         "--n_folds",  str(self.args.n_folds),
#         #         "--logdir",   self.args.logdir,
#         #     ], check=True)

#         # 3) Reload the newly updated model
#         updated_model = torch.load(self.model_path, weights_only=False)
#         new_params = [
#             val.cpu().numpy()
#             for val in updated_model.state_dict().values()
#         ]

#         # 4) Tell the server how many examples you used
#         num_examples = len(self.exp._build_dataset())
#         df = pd.read_csv(os.path.join(self.exp.savedir, "results_single_split.csv"))
#         # compute the average of all “correlation_<project>” columns
#         corr_cols = [c for c in df.columns if c.startswith("correlation_")]
#         avg_corr = df[corr_cols].values.mean()

#         # return it to the server:
#         return new_params, num_examples, {"avg_correlation": float(avg_corr)}
        
#         # return new_params, num_examples, {}    

#     # def evaluate(self, parameters, config):
#     #     # (Optional) you can implement local eval here if you want server‐side metrics
#     #     return 0.0, len(self.exp._build_dataset()), {}

#     def evaluate(self, parameters, config):
#         # 1) Load global weights (similar to fit)
#         model = torch.load(self.model_path, weights_only=False)
#         state_dict = model.state_dict()
#         for k, val in zip(state_dict.keys(), parameters):
#             state_dict[k] = torch.tensor(val)
#         model.load_state_dict(state_dict)

#         # 2) Run a local evaluation pass (e.g. exp.single_run with test‐only)
#         #    or directly call your Experiment.cross_validation() on val/test set
#         report = self.exp.single_run(logdir=self.args.logdir)  # returns DataFrame
#         # Compute a single summary metric (e.g. mean correlation)
#         corr_cols = [c for c in report.columns if c.startswith("correlation_")]
#         avg_corr = report[corr_cols].values.mean()

#         # 3) Return (loss, num_examples, {"correlation": avg_corr})
#         #    Flower expects signature: (loss, num_examples, metrics_dict)
#         #    You can set loss=0.0 if you only care about metrics.
#         num_examples = len(self.exp._build_dataset())
#         return 0.0, num_examples, {"avg_correlation": float(avg_corr)}


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--server_address",             type=str, default="127.0.0.1:8080")
#     parser.add_argument("--path_to_slides",             type=str, required=True)
#     parser.add_argument("--tile_coordinates",           type=str, required=True)
#     parser.add_argument("--path_to_save_features",      type=str, required=True)
#     parser.add_argument("--path_to_transcriptome",      type=str, default="data/TCGA_transcriptome/all_transcriptomes.csv")
#     parser.add_argument("--path_to_save_processed_data",type=str, default="data/TCGA_100_supertiles.h5")
#     parser.add_argument("--n_tiles",                    type=int, default=100)
#     parser.add_argument("--config",                     type=str, default="configs/config_all_genes_test.ini")
#     parser.add_argument("--run",                         type=str, choices=["single_run","cross_validation"], default="single_run")
#     parser.add_argument("--n_folds",                     type=int, default=5)
#     parser.add_argument("--logdir",                     type=str, default="./exp")
#     args = parser.parse_args()

#     client = HE2RNAClient(args)
#     fl.client.start_numpy_client(
#         server_address=args.server_address,
#         client=client,
#     )

import argparse
import subprocess

import flwr as fl
import torch
import os
import pandas as pd
from main import Experiment  # your existing Experiment class
from model import HE2RNA
from torch.nn.modules.conv import Conv1d
torch.serialization.add_safe_globals([HE2RNA, Conv1d])
import numpy as np

class HE2RNAClient(fl.client.NumPyClient):
    def __init__(self, args):
        self.args = args

        # 1⃣ Extract ResNet tile features locally
        # subprocess.run(
        #     [
        #         "python",
        #         "extract_tile_features_from_slides.py",
        #         "--path_to_slides", args.path_to_slides,
        #         "--tile_coordinates", args.tile_coordinates,
        #         "--path_to_save_features", args.path_to_save_features,
        #     ],
        #     check=True,
        # )

        # 2⃣ (Optional) any transcriptome_data.py preprocessing step
        # subprocess.run(["python", "transcriptome_data.py"], check=True)

        # 3⃣ Build supertiles locally
        # subprocess.run(
        #     [
        #         "python",
        #         "supertile_preprocessing.py",
        #         "--path_to_transcriptome",           args.path_to_transcriptome,
        #         "--path_to_save_processed_data",     args.path_to_save_processed_data,
        #         "--n_tiles",                         str(args.n_tiles),
        #     ],
        #     check=True,
        # )

        # 4⃣ Initialize your Experiment object (reads config, but does _not_ train yet)
        self.exp = Experiment(args.config)
        print("line 45 executed in fl_client.py")
        self.model_path = os.path.join("all_genes", "model.pt")  
        print("line 47 executed in fl_client.py")

        # → resolves to "all_genes/model.pt"
        
    def _safe_load_model(self):
        """
        Try to torch.load(model.pt). On failure, delete model.pt
        and regenerate a fresh checkpoint via single_run().
        Returns the loaded model.
        """
        try:
            # In PyTorch ≥2.6, default weights_only=True loads a state_dict.
            # But we explicitly allow full-model unpickling if that's what you saved.
            model = torch.load(self.model_path, weights_only=False)
            return model
        except Exception as e:
            print(f"⚠️  Error loading '{self.model_path}': {e}")
            print("→ Removing corrupt checkpoint and regenerating from scratch.")
            try:
                os.remove(self.model_path)
            except OSError:
                pass

            # Run your single‐run to recreate model.pt
            self.exp.single_run(logdir=self.args.logdir)

            # Load again, this time it should succeed
            model = torch.load(self.model_path, weights_only=False)
            return model
        
    def get_parameters(self,config):
        # Return initial model weights as a list of NumPy arrays
        # params = [
        #     val.cpu().numpy()
        #     for val in self.exp.single_run.__self__.model.state_dict().values()
        # ]
        # return params
    #     """Return current global model weights from all_genes/model.pt."""
    #     # If no model exists yet, create it by running one single_run
        if not os.path.exists(self.model_path):
            print("line 87 executed in fl_client.py")

            self.exp.single_run(logdir=self.args.logdir)
        # model = torch.load(self.model_path,weights_only=False)
        model = self._safe_load_model()
        print("line 92 executed in fl_client.py")


        return [val.cpu().numpy() for val in model.state_dict().values()]


    def fit(self, parameters, config):
    #     # 1) Load round‑zero parameters into your model
    #     state_dict = self.exp.single_run.__self__.model.state_dict()
    #     for k, new_val in zip(state_dict.keys(), parameters):
    #         state_dict[k] = torch.tensor(new_val)
    #     self.exp.single_run.__self__.model.load_state_dict(state_dict)

    #     # 2) Run a local training epoch (or full single_run)
    #     self.exp.single_run(logdir=args.logdir)

    #     # 3) Extract updated weights
    #     new_params = [
    #         val.cpu().numpy()
    #         for val in self.exp.single_run.__self__.model.state_dict().values()
    #     ]

    #     # 4) Return them plus number of training examples
    #     num_examples = len(self.exp._build_dataset())
    #     return new_params, num_examples, {}
        """Load global, train locally via main.py CLI, return updated weights."""
        # 1) Overwrite local model with global weights
        if os.path.exists(self.model_path):
            print("line 120 executed in fl_client.py")

            model = torch.load(self.model_path,weights_only=False)
            state_dict = model.state_dict()
            for k, new_val in zip(state_dict.keys(), parameters):
                state_dict[k] = torch.tensor(new_val)
            model.load_state_dict(state_dict)
            torch.save(model, self.model_path)

        # 2) Run your two CLI commands
        # if self.args.run == "single_run":
        #     subprocess.run([
        #         "python", "main.py",
        #         "--config", self.args.config,
        #         "--run",    "single_run",
        #         "--logdir", self.args.logdir,
        #     ], check=True)
        # else:
        #     subprocess.run([
        #         "python", "main.py",
        #         "--config",   self.args.config,
        #         "--run",      "cross_validation",
        #         "--n_folds",  str(self.args.n_folds),
        #         "--logdir",   self.args.logdir,
        #     ], check=True)

        # 3) Reload the newly updated model
        updated_model = torch.load(self.model_path, weights_only=False)
        new_params = [
            val.cpu().numpy()
            for val in updated_model.state_dict().values()
        ]

        # 4) Tell the server how many examples you used
        num_examples = len(self.exp._build_dataset())
        # load the CSV your single_run just wrote

        # Load the training‐split results CSV your single_run just wrote:
        df_train = pd.read_csv(os.path.join(self.exp.savedir, "results_single_split.csv"))
        corr_cols = [c for c in df_train.columns if c.startswith("correlation_")]
        if not corr_cols:
            avg_corr = 0.0
        else:
            avg_corr= float(np.nanmean(df_train[corr_cols].values))

        # Then return it:
        # return new_params, num_examples, {"avg_correlation": avg_corr_train}
        # df = pd.read_csv(os.path.join(self.exp.savedir, "results_single_split.csv"))
        # compute the average of all “correlation_<project>” columns
        # corr_cols = [c for c in df.columns if c.startswith("correlation_")]
        # avg_corr = df[corr_cols].values.mean()

        # return it to the server:
        return new_params, num_examples, {"avg_correlation": float(avg_corr)}
        # return new_params, num_examples, {}    
    

    # def evaluate(self, parameters, config):
    #     # (Optional) you can implement local eval here if you want server‐side metrics
    #     return 0.0, len(self.exp._build_dataset()), {}
    
    def evaluate(self, parameters, config):
        # 1) Load global weights (similar to fit)
        model = torch.load(self.model_path, weights_only=False)
        state_dict = model.state_dict()
        for k, val in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(val)
        model.load_state_dict(state_dict)

        # 2) Run a local evaluation pass (e.g. exp.single_run with test‐only)
        #    or directly call your Experiment.cross_validation() on val/test set
        print("line 177 executed in fl_client.py")
        # report = self.exp.single_run(logdir=self.args.logdir)  # returns DataFrame
        # Compute a single summary metric (e.g. mean correlation)
        report=pd.read_csv("all_genes/results_single_split.csv")
        corr_cols = [c for c in report.columns if c.startswith("correlation_")]
        if not corr_cols:
            raise RuntimeError(
                f"No correlation columns found in {self.exp.savedir}/results_single_split.csv"
            )

        # Extract the matrix of shape (#genes, #projects)
        corr_vals = report[corr_cols].values

        # Compute the average, skipping any NaNs just in case:
        avg_corr = float(np.nanmean(corr_vals))
        # avg_corr = report[corr_cols].values.mean()

        # 3) Return (loss, num_examples, {"correlation": avg_corr})
        #    Flower expects signature: (loss, num_examples, metrics_dict)
        #    You can set loss=0.0 if you only care about metrics.
        num_examples = len(self.exp._build_dataset())
        return 0.0, num_examples, {"avg_correlation": float(avg_corr)}
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_address",             type=str, default="127.0.0.1:8080")
    parser.add_argument("--path_to_slides",             type=str, required=True)
    parser.add_argument("--tile_coordinates",           type=str, required=True)
    parser.add_argument("--path_to_save_features",      type=str, required=True)
    parser.add_argument("--path_to_transcriptome",      type=str, default="data/TCGA_transcriptome/all_transcriptomes.csv")
    parser.add_argument("--path_to_save_processed_data",type=str, default="data/TCGA_100_supertiles.h5")
    parser.add_argument("--n_tiles",                    type=int, default=100)
    parser.add_argument("--config",                     type=str, default="configs/config_all_genes_test.ini")
    parser.add_argument("--run",                         type=str, choices=["single_run","cross_validation"], default="single_run")
    parser.add_argument("--n_folds",                     type=int, default=5)
    parser.add_argument("--logdir",                     type=str, default="./exp")
    args = parser.parse_args()

    client = HE2RNAClient(args)
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
    )