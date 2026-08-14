"""Low-rank adaptation (LoRA) for the EGNN backbone.

`lora_rank` and `lora_alpha` have appeared in the configs since the arm-D runs,
but nothing ever read them -- there was no LoRA anywhere in the tree, so arm D
was full backbone fine-tuning under a misleading name
(`MODIFICATIONS.md`, docs/experiment-plan.md). This is the actual
implementation.

It exists because of a specific constraint. The ATOMICA critic
(`atomica_interface/critic.py`) is a *loss*, not a conditioning input, so the
critic arm deliberately runs with the ATOMICA adapter switched off -- the cached
`pocket_atomica_embeddings` derive from a two-segment encoding containing the
reference ligand and cannot be a sampling-time input. With the adapter gone and
the backbone frozen, nothing has gradients and the critic has nowhere to push.
LoRA gives it somewhere: a small number of trainable parameters that let the
EGNN's feature processing adapt without touching pretrained weights.

Standard formulation. For a frozen `W: in -> out`, the layer computes

    y = W x + (alpha / r) * B (A x)

with `A: in -> r` initialised from a normal distribution and `B: r -> out`
initialised to **zero**, so the adapted model is exactly the pretrained model at
step 0 and training starts from the checkpoint rather than from a perturbation
of it. Only `A` and `B` are trainable.
"""

import math
import re
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """A frozen `nn.Linear` plus a trainable low-rank update.

    Wraps rather than replaces the original layer, so `state_dict` keys for the
    pretrained weights keep their names under `base.` and a LoRA checkpoint
    still identifies which tensor it adapts.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")

        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # A is Kaiming-uniform as in the original LoRA; B stays zero so the
        # product is zero at initialisation and the wrapped model reproduces the
        # checkpoint exactly on the first forward pass.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        update = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return self.base(x) + self.scaling * update

    def extra_repr(self):
        return (f"rank={self.rank}, alpha={self.alpha}, "
                f"in={self.base.in_features}, out={self.base.out_features}")


# Which linear layers to adapt, by qualified-name substring. These are the
# EGNN's per-layer feature transforms (`GCL.edge_mlp`, `GCL.node_mlp`) and the
# coordinate update (`EquivariantUpdate.coord_mlp`) -- the parts that decide how
# features are mixed and where atoms move, which is what the critic's gradient
# is about. `embedding` / `embedding_out` are left frozen: they define the
# interface to the atom-type vocabulary, and adapting them invites drift in
# what the features *mean* rather than in how they are processed.
DEFAULT_TARGETS = ("edge_mlp", "node_mlp", "coord_mlp")


def inject_lora(module: nn.Module, rank: int, alpha: float = 16.0,
                targets: Optional[Sequence[str]] = None,
                dropout: float = 0.0,
                exclude: Iterable[str] = ()) -> int:
    """Replace matching `nn.Linear` layers with `LoRALinear`, in place.

    Returns the number of layers adapted. Matching is on the qualified module
    name, so `targets=("node_mlp",)` adapts every `node_mlp` linear at any depth.
    """
    targets = tuple(targets) if targets is not None else DEFAULT_TARGETS
    exclude = tuple(exclude)
    replaced = 0

    def matches(name: str) -> bool:
        if any(re.search(pattern, name) for pattern in exclude):
            return False
        return any(pattern in name for pattern in targets)

    for parent_name, parent in list(module.named_modules()):
        for child_name, child in list(parent.named_children()):
            qualified = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Linear) and matches(qualified):
                setattr(parent, child_name, LoRALinear(child, rank, alpha, dropout))
                replaced += 1
    return replaced


def mark_only_lora_trainable(module: nn.Module) -> None:
    """Freeze everything except LoRA factors."""
    for name, param in module.named_parameters():
        param.requires_grad_("lora_A" in name or "lora_B" in name)


def lora_parameters(module: nn.Module):
    """The trainable LoRA factors, for building an optimizer param group."""
    return [p for n, p in module.named_parameters()
            if ("lora_A" in n or "lora_B" in n) and p.requires_grad]


def lora_state_dict(module: nn.Module) -> dict:
    """Just the LoRA factors — a few MB instead of a full checkpoint."""
    return {n: p.detach().cpu() for n, p in module.named_parameters()
            if "lora_A" in n or "lora_B" in n}


def summarize(module: nn.Module) -> str:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    adapted = sum(1 for m in module.modules() if isinstance(m, LoRALinear))
    pct = 100.0 * trainable / total if total else 0.0
    return (f"{adapted} LoRA layers, {trainable:,} trainable of {total:,} "
            f"parameters ({pct:.2f}%)")
