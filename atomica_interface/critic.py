"""ATOMICA as a frozen training-time critic for the diffusion model.

The objective (docs/experiment-plan.md, "The current direction"):

    L = L_diffusion + lambda * d( ATOMICA(pocket, x0_hat), ATOMICA(pocket, x_true) )

`x0_hat` is the denoiser's predicted clean ligand at step `t`, `x_true` is the
reference ligand, and ATOMICA never updates. This only ever asks ATOMICA to
compare two states of the *same* complex, which is the Phase 0 regime measured
at AUROC 1.000 -- never for a transferable absolute score across pockets, which
is the Phase 2 regime that failed. Using `x_true` in a loss is a label, not
leakage; leakage would be feeding it as a conditioning input at sampling time.

## What makes this differentiable

ATOMICA consumes a record of `(X, A, B, block_lengths, segment_ids)`, of which
only `X` is geometric. Block identity, block length, atom type and segment
membership all come from the molecular graph -- the amino-acid sequence of the
pocket and the PS_300 fragmentation of the ligand -- and none of them depends on
coordinates. So the whole non-differentiable half of the featurization can be
computed once, offline, and cached per complex
(`scripts/add_critic_targets.py`); at training time this module only scatters
coordinates into a tensor and runs the encoder, and autograd reaches `x0_hat`
through it.

Two coordinate details are easy to get wrong and are handled explicitly:

- `blocks_to_data` prepends a synthetic global atom per segment whose coordinate
  is the **mean of that segment's atoms**. For the ligand segment that mean
  depends on `x0_hat`, so it is recomputed here rather than cached, or the
  critic would be differentiating through a stale centroid.
- Fragmentation reorders ligand atoms into fragment blocks, so the record's
  ligand rows are a permutation of the stored `lig_coords` rows. The permutation
  is cached as `lig_atom_order` and applied here. Getting this wrong misaligns
  every atom silently, with plausible-looking values -- the same failure mode
  the preprocessing assert guards against.

`interface_data` must have been built with `trim=False`. Trimming selects blocks
by distance, which would make the block structure coordinate-dependent and
invalidate the caching this module relies on.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class ATOMICACritic(nn.Module):
    """Frozen ATOMICA encoder exposing a differentiable interface distance.

    Args:
        config_path / weights_path: the pretrained ATOMICA checkpoint.
        distance: ``cosine`` on the graph representation (the probe found the
            graph level least degenerate for two-segment records) or ``l2`` on
            pooled unit representations.
        level: ``graph`` or ``unit``. ``unit`` is mean-pooled per complex.
    """

    def __init__(self, config_path: str, weights_path: str,
                 distance: str = "cosine", level: str = "graph"):
        super().__init__()
        if distance not in ("cosine", "l2"):
            raise ValueError(f"unknown distance {distance!r}")
        if level not in ("graph", "unit"):
            raise ValueError(f"unknown level {level!r}")

        from atomica_interface.scoring import load_encoder

        self.encoder = load_encoder(config_path, weights_path, device="cpu")
        # Frozen teacher: no parameter updates, no gradient accumulation, and
        # eval mode so any dropout/norm behaves deterministically. Gradients
        # still flow *through* it to the coordinates, which is the point.
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad_(False)

        self.distance = distance
        self.level = level

    def train(self, mode: bool = True):
        """Keep the encoder in eval mode regardless of the parent's mode."""
        super().train(mode)
        self.encoder.eval()
        return self

    @staticmethod
    def assemble_coords(meta: Dict[str, torch.Tensor],
                        pocket_coords: torch.Tensor,
                        lig_coords: torch.Tensor) -> torch.Tensor:
        """Scatter pocket and ligand coordinates into one record's ``X``.

        Row layout, fixed by ``blocks_to_data``::

            [pocket_global, pocket_atoms ..., ligand_global, ligand_atoms ...]

        Only ``lig_coords`` carries gradient; everything else is constant.
        """
        pocket_rows = pocket_coords[meta["pocket_atom_order"]]
        ligand_rows = lig_coords[meta["lig_atom_order"]]
        return torch.cat([
            pocket_rows.mean(dim=0, keepdim=True),   # synthetic global atom
            pocket_rows,
            ligand_rows.mean(dim=0, keepdim=True),   # depends on x0_hat
            ligand_rows,
        ], dim=0)

    def encode(self, metas: List[Dict[str, torch.Tensor]],
               pocket_coords: List[torch.Tensor],
               lig_coords: List[torch.Tensor]) -> torch.Tensor:
        """Encode a batch of complexes, returning one representation per complex."""
        device = lig_coords[0].device
        X, A, B, block_lengths, segment_ids, lengths = [], [], [], [], [], []

        for meta, pocket, lig in zip(metas, pocket_coords, lig_coords):
            X.append(self.assemble_coords(meta, pocket, lig))
            A.append(meta["A"])
            B.append(meta["B"])
            block_lengths.append(meta["block_lengths"])
            segment_ids.append(meta["segment_ids"])
            lengths.append(len(meta["B"]))

        batch = {
            "X": torch.cat(X, dim=0),
            "A": torch.cat(A, dim=0).to(device),
            "B": torch.cat(B, dim=0).to(device),
            "block_lengths": torch.cat(block_lengths, dim=0).to(device),
            "segment_ids": torch.cat(segment_ids, dim=0).to(device),
            "lengths": torch.tensor(lengths, dtype=torch.long, device=device),
        }

        out = self.encoder.infer(batch)
        if self.level == "graph":
            return out.graph_repr

        # Unit level: mean-pool each complex's atoms. `block_lengths` sums to the
        # atom count per complex, so the per-complex atom counts come from
        # splitting it by `lengths`.
        atom_counts = torch.split(batch["block_lengths"], lengths)
        sizes = [int(c.sum()) for c in atom_counts]
        return torch.stack([u.mean(dim=0) for u in torch.split(out.unit_repr, sizes)])

    def pairwise_distance(self, predicted: torch.Tensor,
                          target: torch.Tensor) -> torch.Tensor:
        """Per-complex distance between two batches of representations."""
        if self.distance == "cosine":
            return 1.0 - nn.functional.cosine_similarity(predicted, target, dim=-1)
        return torch.linalg.vector_norm(predicted - target, dim=-1)

    def forward(self, metas: List[Dict[str, torch.Tensor]],
                pocket_coords: List[torch.Tensor],
                lig_coords_hat: List[torch.Tensor],
                target_repr: torch.Tensor) -> torch.Tensor:
        """Critic distance per complex.

        ``target_repr`` is ``ATOMICA(pocket, x_true)``, cached per complex by
        `scripts/add_critic_targets.py`: it is a constant, so the critic costs
        one encoder pass per step rather than two.
        """
        predicted = self.encode(metas, pocket_coords, lig_coords_hat)
        return self.pairwise_distance(predicted, target_repr.to(predicted.dtype))


def lambda_schedule(t_int: torch.Tensor, T: int, max_weight: float,
                    mode: str = "ramp", cutoff: float = 0.5) -> torch.Tensor:
    """Per-sample weight for the critic term as a function of the noise level.

    ATOMICA has never seen a half-formed ligand, and at high `t` the predicted
    `x0_hat` is not a chemically plausible molecule, so weighting every timestep
    equally would spend most of the critic's influence on inputs outside its
    training distribution.

    Args:
        t_int: integer timestep per sample.
        T: total diffusion steps.
        max_weight: lambda at t = 0.
        mode: ``ramp`` decays linearly from ``max_weight`` at t=0 to 0 at
            ``cutoff * T``; ``cutoff`` applies ``max_weight`` below the cutoff
            and 0 above it; ``constant`` applies it everywhere (for ablation).
        cutoff: fraction of T above which the critic is switched off.
    """
    frac = t_int.float() / T
    if mode == "constant":
        return torch.full_like(frac, max_weight)
    if mode == "cutoff":
        return torch.where(frac <= cutoff, max_weight, 0.0)
    if mode == "ramp":
        return max_weight * torch.clamp(1.0 - frac / cutoff, min=0.0)
    raise ValueError(f"unknown schedule mode {mode!r}")


def unpack_critic_meta(data: Dict, index: int,
                       device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    """Pull one complex's cached record metadata out of a collated batch.

    The metadata is ragged (different block counts per complex), so it is
    carried through the collate function as a list rather than a stacked tensor.
    """
    meta = data["critic_meta"][index]
    if device is None:
        return meta
    return {k: v.to(device) for k, v in meta.items()}
