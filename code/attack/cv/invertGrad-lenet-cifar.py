# %% [markdown]
# # Inverting Gradients - How easy is it to break privacy in federated learning?

# %% [markdown]
# This notebook shows an example for a **single image gradient inversion** as described in "Inverting Gradients - How easy is it to break privacy in federated learning?". The setting is a pretrained ResNet-18 and the federated learning algorithm is **fedSGD**.
# 
# Paper URL: https://proceedings.neurips.cc/paper/2020/hash/c4ede56bbd98819ae6112b20ac6bf145-Abstract.html

# %% [markdown]
# #### Abstract
# The idea of federated learning is to collaboratively train a neural network on a server. Each user receives the current weights of the network and in turns sends parameter updates (gradients) based on local data. This protocol has been designed not only to train neural networks data-efficiently, but also to provide privacy benefits for users, as their input data remains on device and only parameter gradients are shared. But how secure is sharing parameter gradients? Previous attacks have provided a false sense of security, by succeeding only in contrived settings - even for a single image. However, by exploiting a magnitude-invariant loss along with optimization strategies based on adversarial attacks, we show that is is actually possible to faithfully reconstruct images at high resolution from the knowledge of their parameter gradients, and demonstrate that such a break of privacy is possible even for trained deep networks. We analyze the effects of architecture as well as parameters on the difficulty of reconstructing an input image and prove that any input to a fully connected layer can be reconstructed analytically independent of the remaining architecture. Finally we discuss settings encountered in practice and show that even averaging gradients over several iterations or several images does not protect the user's privacy in federated learning applications.

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
sensitivity_path = resolve_sensitivity_dir("classification/sensitivity")
print(sensitivity_path)

# %% [markdown]
# ### Initialize cfg object and system setup:

# %% [markdown]
# This will load the full configuration object. This includes the configuration for the use case and threat model as `cfg.case` and the hyperparameters and implementation of the attack as `cfg.attack`. All parameters can be modified below, or overriden with `overrides=` as if they were cmd-line arguments.

# %%
cfg = breaching.get_config(overrides=["case=6_large_batch_cifar"])

device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
setup

# %% [markdown]
# ### Modify config options here

# %% [markdown]
# You can use `.attribute` access to modify any of these configurations for the attack, or the case:

for cfg_user_idx in [93, 27, 48, 43, 11, 80, 8, 44, 2, 79]:
    # %%
    cfg.case.data.partition="unique-class"
    # cfg.case.data.partition="balanced" # 100 unique CIFAR-100 images
    cfg.case.user.num_data_points = 1
    cfg.case.user.user_idx = cfg_user_idx
    cfg.case.model='lenet100'

    cfg.case.user.provide_labels = False
    cfg.attack.label_strategy = "yin" # also works here, as labels are unique

    # Total variation regularization needs to be smaller on CIFAR-10:
    cfg.attack.regularization.total_variation.scale = 5e-4

    # Reduce max iteration to save time
    # cfg.attack.optim.max_iterations = 1


    # %% [markdown]
    # ### Instantiate all parties

    # %% [markdown]
    # The following lines generate "server, "user" and "attacker" objects and print an overview of their configurations.

    # %%
    model_path = f"./classification/model-lenet.pt"
    user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup, model_path=model_path)

    attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
    breaching.utils.overview(server, user, attacker)

    # %%
    server_payload = server.distribute_payload()
    shared_data, true_user_data = user.compute_local_updates(server_payload)

    # %%
    user.plot(true_user_data)

    # %% [markdown]
    # **Network Info**

    # %%
    num_layer = len(shared_data['gradients'])
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

    # %%
    from sewar.full_ref import msssim, uqi, vifp

    def get_sim_metrics(true_data, reconstructed_data):
        def scale_tensor(d):
            min_val, max_val = d.amin(dim=[2, 3], keepdim=True), d.amax(dim=[2, 3], keepdim=True)
            # print(f'min_val: {min_val} | max_val: {max_val}')
            return (d - min_val) / (max_val - min_val)

        true_data_scaled = scale_tensor(true_data["data"]).squeeze().permute(1, 2, 0).cpu().numpy()
        reconstructed_data_scaled = scale_tensor(reconstructed_data["data"]).squeeze().permute(1, 2, 0).cpu().numpy()

        metrics = {
            "msssim": msssim(true_data_scaled, reconstructed_data_scaled, MAX=1.0),
            "uqi": uqi(true_data_scaled, reconstructed_data_scaled),
            "vifp": vifp(true_data_scaled, reconstructed_data_scaled)
        }
        return metrics

    # %% [markdown]
    # ## No Encryption

    # %%
    results_no_dict = {"mse": [], "msssim": [], "uqi": [], "vifp": []}
    for _ in tqdm(range(10)):
        reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data], {}, dryrun=cfg.dryrun)
        metrics = breaching.analysis.report(reconstructed_user_data, 
                                            true_user_data, [server_payload], 
                                            server.model, order_batch=True, compute_full_iip=False, 
                                            cfg_case=cfg.case, setup=setup)
        sim_score = get_sim_metrics(true_user_data, reconstructed_user_data)
        results_dict_temp = {
            "mse": metrics["mse"],
            "msssim": sim_score["msssim"],
            "uqi": sim_score["uqi"],
            "vifp": sim_score["vifp"]
        }
        for k, v in results_dict_temp.items():
            results_no_dict[k].append(v)
            print(f"{k}: {v}")
        user.plot(reconstructed_user_data)
    safe_torch_save(results_no_dict, f"./classification/results/{cfg.case.user.user_idx}/lenet_cifar_no_protection.pt")

    # %% [markdown]
    # ## Selective Encryption

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

    # %%
    def encrypt_selective(sens_list, ratio_enc, num_repeat=1):
        num_enc = (int) (ratio_enc * num_param)
        encrypt_list = torch.topk(flat_tensor_list(sens_list).abs(), num_enc, largest=True).indices

        results_dict = {"mse": [], "msssim": [], "uqi": [], "vifp": []}
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

            # print(new_shared_data['gradients'])

            new_reconstructed_user_data, stats = attacker.reconstruct([server_payload],
                                                                    [new_shared_data],
                                                                    server.secrets,
                                                                    dryrun=cfg.dryrun)
            metrics = breaching.analysis.report(new_reconstructed_user_data,
                                                true_user_data,
                                                [server_payload],
                                                server.model,
                                                order_batch=True,
                                                compute_full_iip=False,
                                                cfg_case=cfg.case,
                                                setup=setup)
            sim_score = get_sim_metrics(true_user_data, new_reconstructed_user_data)
            results_dict_temp = {
                "mse": metrics["mse"],
                "msssim": sim_score["msssim"],
                "uqi": sim_score["uqi"],
                "vifp": sim_score["vifp"]
            }
            print(f"Encrypted the {ratio_enc * 100:.2f}% parameters")
            for k, v in results_dict_temp.items():
                results_dict[k].append(v)
                print(f"{k}: {v}")
            user.plot(new_reconstructed_user_data)
        return results_dict

    # %% [markdown]
    # ## Random Encryption

    # %%
    # below are for for random encryption by parameters (modified):

    def encrypt_random(ratio_enc, num_repeat=1):
        # Randomly pick a proportion to encrypt
        protected_params = random.sample(range(num_param), (int) (ratio_enc * num_param))

        results_dict = {"mse": [], "msssim": [], "uqi": [], "vifp": []}
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
            metrics = breaching.analysis.report(new_reconstructed_user_data,
                                                true_user_data,
                                                [server_payload],
                                                server.model,
                                                order_batch=True,
                                                compute_full_iip=False,
                                                cfg_case=cfg.case,
                                                setup=setup)
            sim_score = get_sim_metrics(true_user_data, new_reconstructed_user_data)
            results_dict_temp = {
                "mse": metrics["mse"],
                "msssim": sim_score["msssim"],
                "uqi": sim_score["uqi"],
                "vifp": sim_score["vifp"]
            }
            print(f"Randomly encrypted {ratio_enc * 100}% parameters")
            for k, v in results_dict_temp.items():
                results_dict[k].append(v)
                print(f"{k}: {v}")
            user.plot(new_reconstructed_user_data)
        return results_dict

    # %%
    # Load sensitivity
    sens_path = sensitivity_path + f"lenet_mean_sens.pt"
    sens_list = torch.load(sens_path)
    for ratio in [0.001, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3]:
        results_select_dict = encrypt_selective(sens_list, 
                                                ratio_enc=ratio,
                                                num_repeat=10)
        safe_torch_save(results_select_dict,
                f"./classification/results/{cfg.case.user.user_idx}/lenet_cifar_bulwarkfl_{ratio}.pt")

        results_random_list = []
        for i in tqdm(range(1), desc="Testing random init..."):
            results_dict = encrypt_random(ratio_enc=ratio, num_repeat=10)
            results_random_list.append(results_dict)
        safe_torch_save(results_random_list,
                f"./classification/results/{cfg.case.user.user_idx}/lenet_cifar_random_{ratio}.pt")


