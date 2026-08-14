"""Unit tests for the LoRA implementation.

Pure torch -- no ATOMICA, no dataset, no GPU. The properties checked here are
the ones whose failure is silent: a LoRA that is not identity at initialisation
quietly discards the pretrained checkpoint, and a LoRA that leaves base weights
trainable quietly becomes a full fine-tune (which is what arm D turned out to
be, since `lora_rank` was in the configs but read by nothing).

    python tests/test_lora.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "DiffSBDD")))

import torch
import torch.nn as nn

from equivariant_diffusion.lora import (
    LoRALinear,
    inject_lora,
    lora_parameters,
    lora_state_dict,
    mark_only_lora_trainable,
)

failures = []


def check(label, condition, detail=""):
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{detail}")
    if not condition:
        failures.append(label)


def toy_model():
    """A stand-in with the same module names the EGNN uses."""
    model = nn.Module()
    model.edge_mlp = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 8))
    model.node_mlp = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 8))
    model.embedding = nn.Linear(8, 8)  # must NOT be adapted by default
    return model


print("\n--- LoRALinear ---")
torch.manual_seed(0)
base = nn.Linear(8, 4)
lora = LoRALinear(base, rank=2, alpha=16.0)
x = torch.randn(5, 8)

# Identity at initialisation: B is zero, so the wrapped layer must reproduce the
# pretrained layer exactly. Otherwise training starts from a perturbation of the
# checkpoint rather than the checkpoint.
check("identity at init", torch.allclose(lora(x), base(x), atol=1e-7),
      f"   (max diff {float((lora(x) - base(x)).abs().max()):.2e})")
check("output shape preserved", lora(x).shape == (5, 4))
check("base weights frozen", not lora.base.weight.requires_grad)
check("lora_A trainable", lora.lora_A.requires_grad)
check("lora_B trainable", lora.lora_B.requires_grad)
check("lora_B zero at init", float(lora.lora_B.abs().sum()) == 0.0)

# Once B moves off zero the layer must actually change, or the adapter is inert.
with torch.no_grad():
    lora.lora_B.normal_()
check("differs from base once B is non-zero",
      not torch.allclose(lora(x), base(x), atol=1e-6))

print("\n--- injection ---")
model = toy_model()
n = inject_lora(model, rank=4, alpha=8.0)
check("adapted the targeted layers", n == 4, f"   (adapted {n})")
adapted = [name for name, m in model.named_modules() if isinstance(m, LoRALinear)]
check("embedding left alone",
      not any("embedding" in name for name in adapted),
      f"   (adapted: {adapted})")

print("\n--- freezing ---")
mark_only_lora_trainable(model)
trainable = [name for name, p in model.named_parameters() if p.requires_grad]
check("only lora factors trainable",
      all("lora_A" in name or "lora_B" in name for name in trainable),
      f"   ({len(trainable)} tensors)")
check("lora_parameters finds them",
      len(lora_parameters(model)) == len(trainable))
check("state dict holds only lora",
      all("lora_" in k for k in lora_state_dict(model)),
      f"   ({len(lora_state_dict(model))} entries)")

print("\n--- gradients ---")
model.zero_grad()
out = model.edge_mlp(torch.randn(3, 8)).sum()
out.backward()
# At initialisation B = 0, so dL/dA = 0 (it is proportional to B) while
# dL/dB is proportional to A and is non-zero. Exactly half the factors receive
# gradient on the first step; A starts moving once B has.
grads = {name: (p.grad is not None and float(p.grad.abs().sum()) > 0)
         for name, p in model.named_parameters() if p.requires_grad}
b_grads = [v for k, v in grads.items() if "lora_B" in k and "edge_mlp" in k]
check("lora_B receives gradient at step 0", all(b_grads), f"   ({sum(b_grads)} tensors)")
base_grads = [p.grad for name, p in model.named_parameters()
              if "base" in name and p.grad is not None]
check("frozen base weights receive no gradient", not base_grads)

print("\n--- rank validation ---")
try:
    LoRALinear(nn.Linear(4, 4), rank=0)
    check("rejects rank 0", False)
except ValueError:
    check("rejects rank 0", True)

print("\n" + "=" * 55)
if failures:
    print(f"FAILED ({len(failures)}): {', '.join(failures)}")
    sys.exit(1)
print("All LoRA checks passed.")
