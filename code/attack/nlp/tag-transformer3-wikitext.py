# %% [markdown]
# # TAG: Gradient Attack on Transformer-based Language Models

# %% [markdown]
# This notebook shows an example for a **short sentence gradient inversion** as described in "TAG: Gradient Attack on Transformer-based Language Models". The setting is a BERT-base model and the federated learning algorithm is **fedSGD**.
# 
# Paper URL: https://aclanthology.org/2021.findings-emnlp.305/

# %% [markdown]
# #### Abstract
# Although distributed learning has increasingly gained attention in terms of effectively utilizing local devices for data privacy enhancement, recent studies show that publicly shared gradients in the training process can reveal the private training data (gradient leakage) to a third-party. We have, however, no systematic understanding of the gradient leakage mechanism on the Transformer based language models. In this paper, as the first attempt, we formulate the gradient attack problem on the Transformer-based language models and propose a gradient attack algorithm, TAG, to reconstruct the local training data. Experimental results on Transformer, TinyBERT4, TinyBERT6 BERT_BASE, and BERT_LARGE using GLUE benchmark show that compared with DLG, TAG works well on more weight distributions in reconstructing training data and achieves 1.5x recover rate and 2.5x ROUGE-2 over prior methods without the need of ground truth label. TAG can obtain up to 90% data by attacking gradients in CoLA dataset. In addition, TAG is stronger than previous approaches on larger models, smaller dictionary size, and smaller input length. We hope the proposed TAG will shed some light on the privacy leakage problem in Transformer-based NLP models.

# %% [markdown]
# ### Startup

# %%
import os, sys

try:
    sys.path.append(os.getcwd())
    import breaching
except ModuleNotFoundError:
    # You only really need this safety net if you want to run these notebooks directly in the examples directory
    # Don't worry about this if you installed the package or moved the notebook to the main directory.
    import os, sys; os.chdir("../..")
    sys.path.append(os.getcwd())
    import breaching
    
import torch
from tqdm import tqdm
import random, json, copy
import numpy as np
import matplotlib.pyplot as plt

# Redirects logs directly into the jupyter notebook
import logging, sys
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
logger = logging.getLogger()


# =========================
# BulwarkFL helpers (paths + safe save)
# =========================
from pathlib import Path
import torch as _torch

def safe_torch_save(obj, path: str):
    """torch.save with auto mkdir for parent folders."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _torch.save(obj, path)

def _find_project_root(start: Path) -> Path:
    """Find folder that contains 'attack' and 'sensitivity' (or the legacy folders)."""
    for p in [start] + list(start.parents):
        if (p / "attack").exists() and (p / "sensitivity").exists():
            return p
        if (p / "code" / "attack").exists() and (p / "code" / "sensitivity").exists():
            return p / "code"
    return start

def resolve_sensitivity_dir(default_subdir: str) -> str:
    """default_subdir: 'classification/sensitivity' or 'lm/sensitivity' used by the original repo."""
    root = _find_project_root(Path(os.getcwd()).resolve())
    # prefer new layout: <root>/sensitivity
    if (root / "sensitivity").exists():
        return str((root / "sensitivity").resolve()) + os.sep
    # fallback to legacy layout: <root>/<default_subdir>
    legacy = (root.parent / default_subdir) if (root.name == "code") else (root / default_subdir)
    return str(legacy.resolve()) + os.sep

# %%
sensitivity_path = resolve_sensitivity_dir("lm/sensitivity")
print(sensitivity_path)

# %% [markdown]
# ### Initialize cfg object and system setup:

# %% [markdown]
# This will load the full configuration object. This includes the configuration for the use case and threat model as `cfg.case` and the hyperparameters and implementation of the attack as `cfg.attack`. All parameters can be modified below, or overriden with `overrides=` as if they were cmd-line arguments.

# %%
cfg = breaching.get_config(overrides=["case=10_causal_lang_training",  "attack=tag"])
          
device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
setup

# %% [markdown]
# ### Modify config options here

# %% [markdown]
# You can use `.attribute` access to modify any of these configurations for the attack, or the case:

for cfg_user_idx in [1, 19683, 20508, 3163, 24283, 12599, 1613, 26838, 10221, 19930]:
    # %%
    cfg.case.user.num_data_points = 1 # How many sentences?
    cfg.case.user.user_idx = cfg_user_idx # From which user?
    # [19683, 20508, 3163, 24283, 12599, 1613, 26838, 10221, 19930]

    cfg.case.data.shape = [16] # This is the sequence length

    cfg.case.model = "transformer3"

    # cfg.attack.optim.max_iterations = 5000 # Increasing the number of iterations can help this attack
    # cfg.attack.optim.optimizer = "gd"

    # %% [markdown]
    # ### Instantiate all parties

    # %% [markdown]
    # The following lines generate "server, "user" and "attacker" objects and print an overview of their configurations.

    # %%
    user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
    model_path = f"./lm/causal-lm/model-tag-{cfg.case.model}.pt"
    model.load_state_dict(torch.load(model_path))

    attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
    breaching.utils.overview(server, user, attacker)

    # %% [markdown]
    # ### Simulate an attacked FL protocol

    # %% [markdown]
    # This exchange is a simulation of a single query in a federated learning protocol. The server sends out a `server_payload` and the user computes an update based on their private local data. This user update is `shared_data` and contains, for example, the parameter gradient of the model in the simplest case. `true_user_data` is also returned by `.compute_local_updates`, but of course not forwarded to the server or attacker and only used for (our) analysis.

    # %%
    server_payload = server.distribute_payload()
    shared_data, true_user_data = user.compute_local_updates(server_payload)

    # %%
    user.print(true_user_data)

    # %%
    num_layer = len(model.state_dict())
    print(f"Number of Layers: {num_layer}")

    # %%
    num_param = 0
    num_param_list = []
    for layer in shared_data['gradients']:
        num_param += layer.numel()
        num_param_list.append(layer.numel())
    print(f"Number of Parameters: {num_param}")

    # %%
    num_param_list

    # %% [markdown]
    # ### Reconstruct user data:

    # %% [markdown]
    # Now we launch the attack, reconstructing user data based on only the `server_payload` and the `shared_data`. 
    # 
    # You can interrupt the computation early to see a partial solution.

    # %%
    results_no_dict = {
        "accuracy": [], 
        "sacrebleu": [], 
        "feat_mse": [], 
        "google_bleu": [],
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
        "token_acc": [],
        "token_avg_accuracy": []
    }
    for i in tqdm(range(10)):
        print(f"===================={i}=========================")
        reconstructed_user_data, stats = attacker.reconstruct([server_payload],
                                                            [copy.deepcopy(shared_data)],
                                                            server.secrets, 
                                                            dryrun=cfg.dryrun)

        metrics = breaching.analysis.report(reconstructed_user_data,
                                            true_user_data,
                                            [server_payload], 
                                            server.model,
                                            order_batch=True,
                                            compute_full_iip=False, 
                                            cfg_case=cfg.case,
                                            setup=setup)
        for k in results_no_dict.keys():
            results_no_dict[k].append(metrics[k])
        user.print_and_mark_correct(reconstructed_user_data, true_user_data)
    safe_torch_save(results_no_dict, f"./lm/results/tag/{cfg.case.user.user_idx}/{cfg.case.model}_no_protection.pt")

    # %%
    def flat_tensor_list(grads):
        flat_grads = torch.tensor([], device=grads[0].device)
        for i in grads:
            flat_grads = torch.concat((flat_grads, i.view(-1)), dim = 0)
        return flat_grads

    def encrypt_by_index(grad_list: list, idx_list: float, if_plot=False) -> list:
        """
        @grad_list: gradient list to be encrypted
        @idx_list: list of encrypted parameter indices
        @if_plot: whether to plot the encrypted ratio of each layer
        """
        grad_flat = flat_tensor_list(grad_list)

        mask = torch.zeros(grad_flat.shape).to(device=grad_flat.device) == 0
        mask[idx_list] = False

        # Apply the encryption mask
        grad_flat = grad_flat * mask + torch.rand(grad_flat.shape).to(device=grad_flat.device) * ~mask

        # Reshape back
        new_grad_list = []
        start_idx = 0
        ratio_enc_per_layer = []
        for layer_grad in tqdm(grad_list):
            layer_param_num = layer_grad.numel()
            grad_slice = grad_flat[start_idx: start_idx + layer_param_num]
            new_grad_list.append(grad_slice.reshape(layer_grad.shape))

            # Record the number of encrypted parameters of each layer
            mask_slice = mask[start_idx: start_idx + layer_param_num]
            ratio_param_enc_layer = (layer_param_num - mask_slice.sum()) / layer_param_num
            ratio_enc_per_layer.append(ratio_param_enc_layer.item())

            start_idx += layer_param_num

        if if_plot:
            plt.bar(np.arange(1, len(ratio_enc_per_layer)+1), ratio_enc_per_layer)
            plt.xlabel("Layer Index")
            plt.ylabel("Encryption Ratio")
            plt.show()
        return new_grad_list

    # %% [markdown]
    # ### Selective Encryption by Sensitivity

    # %%
    def encrypt_selective(sens_list, ratio_enc, num_repeat=1):
        num_enc = (int) (ratio_enc * num_param)
        encrypt_list = torch.topk(flat_tensor_list(sens_list).abs(), num_enc, largest=True).indices

        results_dict = {
            "accuracy": [], 
            "sacrebleu": [], 
            "feat_mse": [], 
            "google_bleu": [],
            "rouge1": [],
            "rouge2": [],
            "rougeL": [],
            "token_acc": [],
            "token_avg_accuracy": []
        }
        for _ in tqdm(range(num_repeat), desc=f"p={ratio_enc}"):
            new_gradient = copy.deepcopy(shared_data['gradients'])
            new_gradient = encrypt_by_index(new_gradient, encrypt_list)

            gradient_with_he = list(new_gradient)
            new_shared_data = dict()
            for k in shared_data.keys():
                if k == "gradients":
                    new_shared_data[k] = gradient_with_he
                else:
                    new_shared_data[k] = shared_data[k]

            new_reconstructed_user_data, stats = attacker.reconstruct([server_payload],
                                                                    [new_shared_data],
                                                                    server.secrets,
                                                                    dryrun=cfg.dryrun)
            print(f"Encrypted the {ratio_enc * 100:.2f}% parameters")
            metrics = breaching.analysis.report(new_reconstructed_user_data,
                                                true_user_data,
                                                [server_payload],
                                                server.model,
                                                order_batch=True,
                                                compute_full_iip=False,
                                                cfg_case=cfg.case,
                                                setup=setup)
            for k in results_dict.keys():
                results_dict[k].append(metrics[k])
            user.print_and_mark_correct(new_reconstructed_user_data, true_user_data)
        return results_dict
            

    # %%
    sens_path = sensitivity_path + f"{cfg.case.model}_tag_mean_sens.pt"
    sens_list = torch.load(sens_path)

    for ratio in [0.05, 0.1, 0.2, 0.3, 0.5]:
        results_select_dict = encrypt_selective(sens_list, 
                                                ratio_enc=ratio,
                                                num_repeat=10)
        safe_torch_save(results_select_dict, f"./lm/results/tag/{cfg.case.user.user_idx}/{cfg.case.model}_bulwarkfl_{ratio}.pt")

    # %%
    # below are for for random encryption by parameters (modified):

    def encrypt_random(ratio_enc, num_repeat=1):
        # Randomly pick a proportion to encrypt
        protected_params = random.sample(range(num_param), (int) (ratio_enc * num_param))

        results_dict = {
            "accuracy": [], 
            "sacrebleu": [], 
            "feat_mse": [], 
            "google_bleu": [],
            "rouge1": [],
            "rouge2": [],
            "rougeL": [],
            "token_acc": [],
            "token_avg_accuracy": []
        }

        for _ in tqdm(range(num_repeat), desc=f"p={ratio_enc}"):
            new_gradient = copy.deepcopy(shared_data['gradients'])
            new_gradient = encrypt_by_index(new_gradient, protected_params)

            gradient_with_he = list(new_gradient)
            new_shared_data = dict()
            for k in shared_data.keys():
                if k == "gradients":
                    new_shared_data[k] = gradient_with_he
                else:
                    new_shared_data[k] = shared_data[k]

            # print(new_shared_data['gradients'])

            new_reconstructed_user_data, stats = attacker.reconstruct([server_payload],
                                                                    [new_shared_data],
                                                                    server.secrets,
                                                                    dryrun=cfg.dryrun)
            print(f"Randomly encrypted {ratio_enc * 100}% parameters")
            metrics = breaching.analysis.report(new_reconstructed_user_data,
                                                true_user_data,
                                                [server_payload],
                                                server.model,
                                                order_batch=True,
                                                compute_full_iip=False,
                                                cfg_case=cfg.case,
                                                setup=setup)
            
            for k in results_dict.keys():
                results_dict[k].append(metrics[k])
            user.print_and_mark_correct(new_reconstructed_user_data, true_user_data)
        return results_dict

    # %%
    for ratio in [0.05, 0.1, 0.2, 0.3, 0.5]:
        results_random_list = []
        for _ in tqdm(range(10), desc="Testing random init..."):
            results_dict = encrypt_random(ratio_enc=ratio, num_repeat=10)
            results_random_list.append(results_dict)
        safe_torch_save(results_random_list, f"./lm/results/tag/{cfg.case.user.user_idx}/{cfg.case.model}_random_{ratio}.pt")


