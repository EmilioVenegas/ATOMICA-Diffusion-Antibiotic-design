import argparse
from argparse import Namespace
from pathlib import Path
import warnings

import torch
torch.set_float32_matmul_precision('medium')
torch.set_float32_matmul_precision('medium')
import pytorch_lightning as pl
import yaml
import numpy as np

from lightning_modules import LigandPocketDDPM


def merge_args_and_yaml(args, config_dict):
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if key in arg_dict:
            warnings.warn(f"Command line argument '{key}' (value: "
                          f"{arg_dict[key]}) will be overwritten with value "
                          f"{value} provided in the config file.")
        if isinstance(value, dict):
            arg_dict[key] = Namespace(**value)
        else:
            arg_dict[key] = value

    return args


def merge_configs(config, resume_config):
    """
    Merge resume config with current config.
    """
    # List of keys that should NOT be overwritten by the checkpoint
    # This allows adjusting hardware-dependent parameters (like batch_size) when resuming
    keys_to_keep = {'batch_size', 'num_workers', 'gpus', 'accumulate_grad_batches', 'gradient_clip_val',
                    'auxiliary_loss', 'loss_params', 'bond_params', 'coord_loss_weight', 'eval_params'}

    for key, value in resume_config.items():
        if key in keys_to_keep:
            continue
            
        if isinstance(value, Namespace):
            value = value.__dict__
        
        # Skip keys that we want to preserve from the new config
        if key in ['datadir', 'dataset', 'num_workers', 'batch_size', 'accumulate_grad_batches', 'precision', 
                   'lr', 'adapter_lr', 'freeze_backbone']:
            continue

        if key in config and config[key] != value:
            # Handle nested dictionaries (like egnn_params)
            if isinstance(config[key], dict) and isinstance(value, dict):
                # Update the checkpoint dict with the new config's extra keys (e.g. atomica_nf)
                # But we want to keep the checkpoint's values for existing keys (to match weights)
                # So we take value (checkpoint) and update it with keys from config[key] that are NOT in value
                for k, v in config[key].items():
                    if k not in value:
                        value[k] = v
                        print(f"Preserving new config key '{key}.{k}': {v}")
            
            warnings.warn(f"Config parameter '{key}' (value: "
                          f"{config[key]}) will be overwritten with value "
                          f"{value} from the checkpoint.")
        config[key] = value
    return config


# ------------------------------------------------------------------------------
# Training
# ______________________________________________________________________________
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--resume', type=str, default=None)
    args = p.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    assert 'resume' not in config

    # Get main config
    ckpt_path = None if args.resume is None else Path(args.resume)
    if args.resume is not None:
        resume_config = torch.load(
            ckpt_path, map_location=torch.device('cpu'))['hyper_parameters']

        config = merge_configs(config, resume_config)

    args = merge_args_and_yaml(args, config)

    out_dir = Path(args.logdir, args.run_name)
    # Fail loudly if the histogram file is missing
    histogram_file = Path(args.datadir, 'size_distribution.npy')
    histogram = np.load(histogram_file).tolist()
    pl_module = LigandPocketDDPM(
        outdir=out_dir,
        dataset=args.dataset,
        datadir=args.datadir,
        batch_size=args.batch_size,
        lr=args.lr,
        adapter_lr=getattr(args, 'adapter_lr', args.lr * 0.01),  # Default to 1% of base LR
        freeze_backbone=getattr(args, 'freeze_backbone', False),  # Default to unfrozen
        egnn_params=args.egnn_params,
        diffusion_params=args.diffusion_params,
        num_workers=args.num_workers,
        augment_noise=args.augment_noise,
        augment_rotation=args.augment_rotation,
        clip_grad=args.clip_grad,
        eval_epochs=args.eval_epochs,
        eval_params=args.eval_params,
        visualize_sample_epoch=args.visualize_sample_epoch,
        visualize_chain_epoch=args.visualize_chain_epoch,
        auxiliary_loss=args.auxiliary_loss,
        loss_params=args.loss_params,
        mode=args.mode,
        node_histogram=histogram,
        pocket_representation=args.pocket_representation,
        virtual_nodes=getattr(args, 'virtual_nodes', False)
    )

    logger = pl.loggers.WandbLogger(
        save_dir=args.logdir,
        project=args.wandb_params.project,
        group=args.wandb_params.group,
        name=args.run_name,
        id=args.run_name,
        resume='allow' if args.resume is not None else False,
        entity=args.wandb_params.entity,
        mode=args.wandb_params.mode,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=Path(out_dir, 'checkpoints'),
        filename="best-model-epoch={epoch:02d}",
        monitor="loss/val",
        save_top_k=1,
        save_last=True,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        precision='16-mixed',
        enable_progress_bar=args.enable_progress_bar,
        num_sanity_val_steps=args.num_sanity_val_steps,
        accelerator='gpu', devices=args.gpus,
        strategy=('ddp' if args.gpus > 1 else 'auto'),
        precision=args.precision
    )

    if ckpt_path is not None:
        print(f"Loading pre-trained weights from: {ckpt_path}")
        
        # Check if we are doing transfer learning (missing keys expected) or resuming
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        # Try to load strict to see if it matches
        try:
            pl_module.load_state_dict(state_dict, strict=True)
            is_exact_match = True
        except RuntimeError:
            is_exact_match = False
            
        if is_exact_match:
            print("Checkpoint matches model architecture exactly. Resuming training state (epochs, optimizer)...")
            trainer.fit(model=pl_module, ckpt_path=ckpt_path)
        else:
            print("Transfer learning mode: loading backbone weights, initializing new layers randomly")
            # Load state dict with strict=False to allow missing keys (ATOMICA layers)
            missing_keys, unexpected_keys = pl_module.load_state_dict(
                state_dict, strict=False)
            
            if missing_keys:
                print(f"Initialized {len(missing_keys)} new parameters (ATOMICA layers):")
                for key in missing_keys[:5]:  # Show first 5
                    print(f"  - {key}")
                if len(missing_keys) > 5:
                    print(f"  ... and {len(missing_keys) - 5} more")
            
            # Train from epoch 0 (transfer learning)
            trainer.fit(model=pl_module)
    else:
        trainer.fit(model=pl_module)
