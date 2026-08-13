"""Training-free pose energy from ATOMICA's pretrained denoising heads.

ATOMICA was pretrained to predict the rigid-body noise applied to one segment of
a complex. Feeding a pose and asking how large a correction it predicts therefore
yields a pose score with no fitting at all: a native pose should need almost none,
a displaced pose should need a lot.

The trick is that `translation_loss` is computed as
``mse(tr_eps * trans_noise, -tr_score)``, so passing a **zero** noise target makes
the loss equal the magnitude of the predicted correction. No labels, no probe.

Measured usefulness is limited -- see `results/phase0/README.md`. On the
clash-controlled CDK2 pose benchmark this energy reaches AUROC 0.787 against a
0.727 trivial steric baseline, while a linear probe on the same representation
reaches 1.000. The signal is present but the pretrained heads do not surface it,
so a usable scorer needs a head trained across systems.

Only translation is used. The rotation head inverts on this benchmark (displaced
poses score *lower* than native), so it is reported but not combined.
"""

from typing import Optional

import torch


def load_denoiser(config_path: str, weights_path: str, device: str = "cpu"):
    """Load the pretrained denoiser with the heads we cannot evaluate switched off.

    Torsion scoring needs torsion edges, which a rigid-body pose comparison does
    not define, and the masking objective needs mask labels. Both raise on `None`
    inside `forward`, so they are disabled rather than fed dummy targets.
    """
    from ATOMICA.models.pretrain_model import DenoisePretrainModel

    model = DenoisePretrainModel.load_from_config_and_weights(config_path, weights_path)
    model.torsion_noise = False
    model.masking_objective = False
    return model.to(device).eval()


def pose_energy(
    model,
    batch,
    ligand_segment: int = 1,
    device: Optional[str] = None,
) -> dict:
    """Predicted rigid-body correction magnitude for one complex.

    Lower means the model wants to move the ligand less, i.e. a more native-like
    pose. Use ``-energy`` when a higher-is-better score is wanted.

    ``batch`` is a record from `featurize.to_batch`; ``ligand_segment`` selects
    which segment is treated as mobile (1 = ligand under the pocket-first
    convention of `featurize.interface_data`).
    """
    device = device or batch["X"].device
    zeros3 = torch.zeros(1, 3, device=device)
    empty_long = torch.zeros(2, 0, dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(
            Z=batch["X"],
            B=batch["B"],
            A=batch["A"],
            block_lengths=batch["block_lengths"],
            lengths=batch["lengths"],
            segment_ids=batch["segment_ids"],
            receptor_segment=torch.tensor([ligand_segment], device=device),
            # Zero targets -> the loss reduces to the magnitude of the prediction.
            tr_score=zeros3,
            tr_eps=torch.ones(1, device=device),
            rot_score=zeros3,
            tor_edges=empty_long,
            tor_score=torch.zeros(0, device=device),
            tor_batch=torch.zeros(0, dtype=torch.long, device=device),
        )

    return {
        "translation": float(out.translation_loss),
        "rotation": float(out.rotation_loss),
    }
