import torch
import numpy as np
from pathlib import Path
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation

from src.dataset.BUSI_dataloader import load_datasets
from src.utils.miscellany import load_config_file, seed_everything
from src.utils.experiment_init import device_setup
from src.utils.models import load_pretrained_model
from src.models.multitask.MTUNetPlusPlus import MTUNetPlusPlus

# ── CONFIG ──────────────────────────────────────────────────────────────────
# CHANGE THIS to your actual run folder name
RUN_PATH ="C:\new_mtunet\runs\20260420_160501_MTUNetPlusPlus_24_alpha_0.85_batch_2_benign_malignant_normal"
# ─────────────────────────────────────────────────────────────────────────────

config_model, config_opt, config_loss, config_training, config_data = load_config_file('./src/config.yaml')
seed_everything(config_training['seed'])
dev = device_setup()

transforms = torch.nn.Sequential(
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    RandomRotation(degrees=360)
)
train_loaders, val_loaders, test_loaders = load_datasets(config_training, config_data, transforms, mode='CV')

for fold_n, (train_loader, val_loader, test_loader) in enumerate(zip(train_loaders, val_loaders, test_loaders)):
    print(f"\nExtracting features for fold {fold_n}...")

    # find checkpoint file
    fold_path = Path(RUN_PATH) / f"fold_{fold_n}"
    ckpt_files = list(fold_path.glob("model_*"))
    assert len(ckpt_files) > 0, f"No checkpoint found in {fold_path}"
    ckpt_path = str(ckpt_files[0])
    print(f"  Loading: {ckpt_path}")

    # load model
    model = MTUNetPlusPlus(
        spatial_dims=2,
        in_channels=config_model.get('in_channels', 1),
        out_channels=config_model.get('out_channels', 1),
        n_classes=len(config_data['classes']),
        deep_supervision=False
    ).to(dev)
    model = load_pretrained_model(model, ckpt_path)
    model.eval()

    # hook into classifier[3] = nn.Linear(512, 256) output
    captured = {}
    def hook_fn(module, input, output):
        captured['feat'] = output.detach().cpu()

    hook = model.classifier[3].register_forward_hook(hook_fn)

    def extract(loader, split_name):
        feats, labels = [], []
        with torch.no_grad():
            for data in loader:
                imgs = data['image'].to(dev)
                lbl  = data['label'].cpu()
                model(imgs)
                feats.append(captured['feat'].numpy())
                # label: convert one-hot or int to class index
                if lbl.ndim > 1 and lbl.shape[1] > 1:
                    lbl = torch.argmax(lbl, dim=1)
                labels.append(lbl.numpy().flatten())
        feats  = np.vstack(feats)
        labels = np.concatenate(labels)
        np.save(fold_path / f"{split_name}_features.npy", feats)
        np.save(fold_path / f"{split_name}_labels.npy",   labels)
        print(f"  {split_name}: {feats.shape}, labels: {labels.shape}")

    extract(train_loader, "train")
    extract(test_loader,  "test")
    hook.remove()

print("\nDone. Features saved in each fold folder.")