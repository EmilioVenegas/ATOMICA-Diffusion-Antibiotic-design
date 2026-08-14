"""Pick `critic_params.max_weight` by measuring gradients, not by guessing.

The critic term enters as

    L = L_diffusion + lambda(t) * d( ATOMICA(pocket, x0_hat), ATOMICA(pocket, x_true) )

and the question "how big should lambda be" is really "how much of the update
should the critic drive". Comparing the two *losses* answers the wrong question:
cosine distance on a 32-d representation is intrinsically small (order 1e-2),
while the diffusion nll is order 1e0, so the loss ratio makes the critic look
negligible even at a lambda that would dominate training. What matters is the
norm of the gradient each term contributes to the trainable parameters.

This runs both terms separately over real batches, with lambda factored out, and
reports the lambda that would give the critic a chosen share of the total
gradient norm. Nothing is trained and nothing is written.

Reported per timestep band, because lambda is ramped: the critic is applied at
low `t`, where `x0_hat` is chemically plausible, and the gradient ratio there is
the one that governs.

Usage (from repo root):

    python scripts/calibrate_critic_weight.py --data_dir data/processed_expert_atomica/val
"""

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "DiffSBDD")))

import numpy as np
import torch
import yaml

from DiffSBDD.dataset import LigandPocketDatasetPT
from DiffSBDD.lightning_modules import LigandPocketDDPM


def grad_norm(module, loss, retain=False):
    """L2 norm of dloss/dtheta over trainable parameters, without stepping."""
    params = [p for p in module.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=retain,
                                allow_unused=True)
    total = 0.0
    for g in grads:
        if g is not None:
            total += float(g.pow(2).sum())
    return total ** 0.5


def build_module(cfg, datadir, histogram, device):
    module = LigandPocketDDPM(
        outdir=Path("/tmp/calibrate"),
        dataset=cfg["dataset"],
        datadir=datadir,
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        adapter_lr=cfg["adapter_lr"],
        freeze_backbone=cfg["freeze_backbone"],
        egnn_params=Namespace(**cfg["egnn_params"]),
        diffusion_params=Namespace(**cfg["diffusion_params"]),
        num_workers=0,
        augment_noise=cfg["augment_noise"],
        augment_rotation=cfg["augment_rotation"],
        clip_grad=cfg["clip_grad"],
        eval_epochs=cfg["eval_epochs"],
        eval_params=Namespace(**{**cfg["eval_params"], "smiles_file": None}),
        visualize_sample_epoch=cfg["visualize_sample_epoch"],
        visualize_chain_epoch=cfg["visualize_chain_epoch"],
        auxiliary_loss=cfg["auxiliary_loss"],
        loss_params=Namespace(**cfg["loss_params"]),
        mode=cfg["mode"],
        node_histogram=histogram,
        pocket_representation=cfg["pocket_representation"],
        critic_params=Namespace(**cfg["critic_params"]),
    )
    return module.to(device)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="DiffSBDD/configs/crossdock_fullatom_critic.yml")
    p.add_argument("--data_dir", default="data/processed_expert_atomica/val")
    p.add_argument("--batches", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(args.config))
    cfg["egnn_params"]["device"] = str(device)
    cfg["egnn_params"]["gradient_checkpointing"] = False
    # Factor lambda out: measure the bare critic gradient and scale afterwards.
    cfg["critic_params"]["max_weight"] = 1.0
    cfg["critic_params"]["schedule"] = "constant"

    histogram = np.load(Path(args.data_dir).parent / "size_distribution.npy").tolist()
    module = build_module(cfg, args.data_dir, histogram, device)
    module.ddpm.train()

    dataset = LigandPocketDatasetPT(args.data_dir)
    augmented = [i for i in range(len(dataset)) if dataset[i]["critic_meta"] is not None]
    print(f"{len(dataset)} complexes, {len(augmented)} carrying critic targets")
    if not augmented:
        print("No complexes carry critic targets; run scripts/add_critic_targets.py.")
        return

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # Bands over the diffusion timestep, since lambda is ramped and only the low
    # band is actually weighted in training.
    bands = {"t/T <= 0.25": [], "0.25 < t/T <= 0.5": [], "t/T > 0.5": []}
    ratios, diffusion_norms, critic_norms = [], [], []

    for _ in range(args.batches):
        idx = rng.choice(augmented, size=min(args.batch_size, len(augmented)),
                         replace=False)
        batch = LigandPocketDatasetPT.collate_fn([dataset[int(i)] for i in idx])

        # Diffusion-only gradient.
        module.critic, saved = None, module.critic
        module.zero_grad()
        nll_plain, _ = module.forward(batch)
        g_diffusion = grad_norm(module, nll_plain.mean())
        module.critic = saved

        # Critic-only gradient, lambda = 1.
        module.zero_grad()
        ligand, pocket = module.get_ligand_and_pocket(batch)
        (delta_log_px, error_t_lig, error_t_pocket, SNR_weight,
         loss_0_x_ligand, loss_0_x_pocket, loss_0_h, neg_log_const_0,
         kl_prior, log_pN, t_int, xh_lig_hat, info) = module.ddpm(
            ligand, pocket, return_info=True)
        xh_pocket = info.pop("_xh_pocket")
        weighted, critic_info = module.critic_term(
            batch, ligand, pocket, xh_lig_hat, xh_pocket, t_int)
        if weighted is None:
            continue
        g_critic = grad_norm(module, weighted.mean())

        if g_critic == 0:
            continue
        ratio = g_diffusion / g_critic
        ratios.append(ratio)
        diffusion_norms.append(g_diffusion)
        critic_norms.append(g_critic)

        frac = float(t_int.float().mean()) / module.T
        band = ("t/T <= 0.25" if frac <= 0.25
                else "0.25 < t/T <= 0.5" if frac <= 0.5 else "t/T > 0.5")
        bands[band].append(ratio)

    if not ratios:
        print("No batch produced a critic gradient.")
        return

    ratios = np.array(ratios)
    print(f"\n{'=' * 70}")
    print(f"GRADIENT CALIBRATION over {len(ratios)} batches of {args.batch_size}")
    print("=" * 70)
    print(f"||grad|| diffusion : {np.mean(diffusion_norms):.4e} "
          f"(sd {np.std(diffusion_norms):.1e})")
    print(f"||grad|| critic    : {np.mean(critic_norms):.4e} "
          f"(sd {np.std(critic_norms):.1e})   [at lambda = 1]")
    print(f"ratio diffusion/critic : {ratios.mean():.1f}  "
          f"median {np.median(ratios):.1f}")

    print(f"\nlambda for a target share of total gradient norm:")
    print(f"{'critic share':>14}  {'lambda':>12}")
    print("-" * 28)
    for share in (0.05, 0.10, 0.25, 0.50):
        # lambda * |g_c| / (|g_d| + lambda * |g_c|) = share
        lam = share / (1 - share) * float(np.median(ratios))
        print(f"{share:>13.0%}  {lam:>12.3g}")

    print(f"\nBy timestep band (lambda is ramped, so the low band governs):")
    for band, values in bands.items():
        if values:
            print(f"  {band:<20} n={len(values):<3} median ratio "
                  f"{np.median(values):>8.1f}  ->  lambda for 10% share: "
                  f"{0.1 / 0.9 * float(np.median(values)):.3g}")
        else:
            print(f"  {band:<20} n=0")

    print("\nThese are gradient norms at the current parameters, not a guarantee "
          "about training dynamics. Start at the 10% figure, then confirm "
          "critic_distance/train actually falls.")


if __name__ == "__main__":
    main()
