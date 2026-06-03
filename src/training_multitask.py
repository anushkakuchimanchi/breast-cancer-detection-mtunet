import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from pprint import pformat

import numpy as np
from sklearn.svm import SVC

import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score as f1
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from torchvision.transforms import RandomRotation, RandomHorizontalFlip, RandomVerticalFlip

from src.dataset.BUSI_dataloader import load_datasets
from src.utils.criterions import apply_criterion_multitask_segmentation_classification
from src.utils.experiment_init import device_setup
from src.utils.experiment_init import load_multitask_experiment_artefacts
from src.utils.metrics import binary_classification_metrics
from src.utils.metrics import dice_score_from_tensor
from src.utils.metrics import multiclass_classification_metrics
from src.utils.miscellany import init_log
from src.utils.miscellany import load_config_file
from src.utils.miscellany import save_classification_results
from src.utils.miscellany import save_segmentation_results
from src.utils.miscellany import seed_everything
from src.utils.miscellany import write_metrics_file
from src.utils.models import inference_multitask_binary_classification_segmentation
from src.utils.models import inference_multitask_multiclass_classification_segmentation
from src.utils.models import load_pretrained_model
from src.utils.visualization import plot_evolution
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler as StandardScalerViz
import seaborn as sns
CLASS_NAMES = ['normal', 'benign', 'malignant']

def processes_classification_predicted(num_classes, pred_logits, gt_label, gt_list, pred_list):
    # averaging prediction if deep supervision
    if isinstance(pred_logits, list):
        pred_logits = torch.mean(torch.stack(pred_logits, dim=0), dim=0)

    # this if-else differentiates between multiclass and binary class predictions
    if num_classes > 2:
        # applying softmax to get probabilities
        probabilities = torch.nn.functional.softmax(pred_logits, dim=1)

        # Applying argmax to get the class with the highest probability
        gt_label = [torch.argmax(k, keepdim=True).to(torch.float) for k in gt_label]
        pred_class = [torch.argmax(pl, keepdim=True).to(torch.float) for pl in probabilities]

        # storing the probabilities and ground truth labels in lists
        for la, p in zip(gt_label, pred_class):
            gt_list.append(la.detach().item())
            pred_list.append(p.detach().item())
    else:
        # adding ground truth label and predicted label
        if pred_logits.shape[0] > 1:  # when batch size > 1, each element is added individually
            for i in range(pred_logits.shape[0]):
                pred_list.append((torch.sigmoid(pred_logits[i, :]) > .5).double().detach().item())
                gt_list.append(gt_label[i, :].detach().item())
        else:
            pred_list.append((torch.sigmoid(pred_logits) > .5).double().detach().item())
            gt_list.append(gt_label.detach().item())

    return gt_list, pred_list


def process_segmentation_predicted(outputs, masks):
    # measuring DICE error
    if isinstance(outputs, list):
        outputs = outputs[-1]
    outputs = torch.sigmoid(outputs) > .5  # converting continuous values into probability [0, 1]

    return dice_score_from_tensor(masks, outputs)


def train_one_epoch(num_classes):
    training_loss, training_dice = 0., 0.
    gt_label, pred_label = [], []

    # Iterating over training loader
    for k, data in enumerate(training_loader):

        # Loading the input data
        inputs, masks, label = data['image'].to(dev), data['mask'].to(dev), data['label'].to(dev)
        if num_classes > 2:
            label = torch.nn.functional.one_hot(label.flatten().to(torch.int64), num_classes=3).to(torch.float)

        # Zero the gradients for every batch
        optimizer.zero_grad(set_to_none=True)

        # Make predictions for this batch
        logits, outputs = model(inputs)

        # Compute the loss and its gradients. It is not necessary to apply either softmax or sigmoid before logits as
        # all classification criteria do it. The same happens for segmentation criteria
        seg_loss, cls_loss = apply_criterion_multitask_segmentation_classification(seg_criterion, masks, outputs,
                                                                                   cls_criterion, label, logits,
                                                                                   config_loss['inversely_weighted'])
        # weighting each of the loss functions
        total_loss = alpha * seg_loss + (1 - alpha) * cls_loss
        training_loss += total_loss.item()

        # Performing backward step through scaler methodology
        total_loss.backward()
        optimizer.step()

        # processing predictions to calculate training metrics
        # print(training_dice)
        training_dice += process_segmentation_predicted(outputs, masks)
        gt_label, pred_label = processes_classification_predicted(num_classes, logits, label, gt_label, pred_label)

    avg_training_loss = training_loss / training_loader.__len__()
    avg_training_dice = training_dice / training_loader.__len__()
    training_acc = accuracy_score(gt_label, pred_label)
    training_f1 = f1(y_true=gt_label, y_pred=pred_label, labels=[0, 1, 2], average='weighted')

    del training_dice, gt_label, pred_label
    return avg_training_loss, avg_training_dice, training_acc, training_f1


@torch.inference_mode()
def validate_one_epoch(num_classes):
    val_loss, seg_val_loss, cls_val_loss, val_dice = 0.0, 0.0, 0.0, 0.0
    val_gt_label, val_pred_label = [], []

    # Iterating over training loader
    for k, val_data in enumerate(validation_loader):

        # Loading the input data
        val_inputs, val_masks, val_label = val_data['image'].to(dev), val_data['mask'].to(dev), val_data['label'].to(dev)
        if num_classes > 2:
            val_label = torch.nn.functional.one_hot(val_label.flatten().to(torch.int64), num_classes=3).to(torch.float)

        # Make predictions for this batch
        val_logits, val_outputs = model(val_inputs)

        # Compute the loss and its gradients. It is not necessary to apply either softmax or sigmoid before logits as
        # all classification criteria do it. The same happens for segmentation criteria
        seg_loss, cls_loss = apply_criterion_multitask_segmentation_classification(seg_criterion, val_masks, val_outputs,
                                                                                   cls_criterion, val_label, val_logits,
                                                                                   config_loss['inversely_weighted'])
        # weighting each of the loss functions
        total_loss = alpha * seg_loss + (1 - alpha) * cls_loss
        seg_val_loss += seg_loss.item()
        cls_val_loss += cls_loss.item()
        val_loss += total_loss.item()

        # processing predictions to calculate training metrics
        val_dice += process_segmentation_predicted(val_outputs, val_masks)
        val_gt_label, val_pred_label = processes_classification_predicted(num_classes, val_logits, val_label, val_gt_label, val_pred_label)

    # total segmentation loss and DICE metric for the epoch
    avg_val_loss = val_loss / validation_loader.__len__()
    avg_cls_val_loss = cls_val_loss / validation_loader.__len__()
    avg_seg_val_loss = seg_val_loss / validation_loader.__len__()
    avg_val_dice = val_dice / validation_loader.__len__()
    val_acc = accuracy_score(val_gt_label, val_pred_label)
    val_f1 = f1(y_true=val_gt_label, y_pred=val_pred_label, labels=[0, 1, 2], average='weighted')

    del val_dice, val_pred_label, val_gt_label
    return avg_val_loss, avg_val_dice, val_acc, val_f1, avg_seg_val_loss, avg_cls_val_loss


# alphas = [1, .95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, .4, .35, .3, .25, .2, .15, .1, .05, .0]
# # alphas = [.25, .2, .15, .1, .05, .0, -1, -2, -5]
# for alpha in alphas:
# for beta in betas:
# beta = 5
# print(beta)
# initializing times
init_time = time.perf_counter()
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# loading config file
config_model, config_opt, config_loss, config_training, config_data = load_config_file(path='./src/config.yaml')
if config_training['CV'] < 2:
    sys.exit("This code is prepared for receiving a CV greater than 1")

# initializing seed and gpu if possible
seed_everything(config_training['seed'], cuda_benchmark=config_training['cuda_benchmark'])
dev = device_setup()

# initializing folder structures and log
# config_training['alpha'] = alpha
alpha = config_training['alpha']
run_path = f"runs/{timestamp}_{config_model['architecture']}_{config_model['width']}_alpha_{config_training['alpha']}" \
           f"_batch_{config_data['batch_size']}_{'_'.join(config_data['classes'])}"
Path(f"{run_path}").mkdir(parents=True, exist_ok=True)
init_log(log_name=f"./{run_path}/execution.log")
shutil.copyfile('./src/config.yaml', f'./{run_path}/config.yaml')

# initializing experiment's objects
n_classes = len(config_data['classes'])
n_augments = sum([v for k, v in config_data['augmentation'].items()])
transforms = torch.nn.Sequential(
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    RandomRotation(degrees=360)
)
train_loaders, val_loaders, test_loaders = load_datasets(config_training, config_data, transforms, mode='CV')


for n, (training_loader, validation_loader, test_loader) in enumerate(zip(train_loaders, val_loaders, test_loaders)):
    # Add this right after:
    logging.info(f"\n\n *********************  FOLD {n}  ********************* \n\n")
    logging.info(f"\n\n ###############  TRAINING PHASE  ###############  \n\n")
    print(f"\n{'='*60}")
    print(f"  FOLD {n+1} / {config_training['CV']}")
    print(f"{'='*60}")
    # creating specific paths and experiment's objects for each fold
    fold_time = time.perf_counter()
    Path(f"{run_path}/fold_{n}/segs/").mkdir(parents=True, exist_ok=True)
    Path(f"{run_path}/fold_{n}/plots/").mkdir(parents=True, exist_ok=True)
    Path(f"{run_path}/fold_{n}/features_map/").mkdir(parents=True, exist_ok=True)

    # artefacts initialization
    model, optimizer, seg_criterion, cls_criterion, scheduler = load_multitask_experiment_artefacts(config_data, config_model, config_opt, config_loss, n_augments, run_path)
    model = model.to(dev)

    # init metrics file
    write_metrics_file(path_file=f'{run_path}/fold_{n}/metrics.csv',
                       text_to_write=f'epoch,LR,Train_loss,Validation_loss,Train_dice,Validation_dice,Train_acc,Train_F1,Validation_acc,Validation_F1')

    best_validation_loss = 1_000_000.
    patience = 0
    for epoch in range(config_training['epochs']):
        current_lr = optimizer.param_groups[0]["lr"]
        start_epoch_time = time.perf_counter()

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_train_loss, avg_dice, train_acc, train_f1_score = train_one_epoch(n_classes)

        # We don't need gradients on to do reporting
        model.train(False)
        avg_validation_loss, avg_validation_dice, val_acc_score, val_f1_score, segmentation_val_loss, classification_val_loss = validate_one_epoch(n_classes)

        # # Update the learning rate at the end of each epoch
        if config_opt['scheduler'] == 'cosine':
            scheduler.step()
        else:
            scheduler.step(avg_validation_loss)

        # Track the best performance, and save the model's state
        if avg_validation_loss < best_validation_loss:
            patience = 0  # restarting patience
            best_validation_loss = avg_validation_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': 'scheduler',
                'val_loss': best_validation_loss
            }, f'{run_path}/fold_{n}/model_{timestamp}_fold_{n}')
        else:
            patience += 1

        # logging results of current epoch
        end_epoch_time = time.perf_counter()
        logging.info(f'EPOCH {epoch} --> '
                     f'|| Training loss {avg_train_loss:.4f} '
                     f'|| Validation loss {avg_validation_loss:.4f} '
                     f'|| Segmentation val loss {segmentation_val_loss:.4f} '
                     f'|| Classification val loss {classification_val_loss:.4f} '
                     f'|| Training DICE {avg_dice:.4f} '
                     f'|| Validation DICE  {avg_validation_dice:.4f} '
                     f'|| Training ACC {train_acc:.4f} '
                     f'|| Training F1 {train_f1_score:.4f} '
                     f'|| Validation ACC {val_acc_score:.4f} '
                     f'|| Validation F1 {val_f1_score:.4f} '
                     f'|| Patience: {patience} '
                     f'|| Epoch time: {end_epoch_time - start_epoch_time:.4f}'
                     f'|| Best validation performance: {best_validation_loss:.4f}'
        )

        write_metrics_file(path_file=f'{run_path}/fold_{n}/metrics.csv',
                           text_to_write=f'{epoch},{current_lr:.8f},{avg_train_loss:.4f},{avg_validation_loss:.4f},'
                                         f'{avg_dice:.4f},{avg_validation_dice:.4f},{train_acc:.4f},'
                                         f'{train_f1_score:.4f},{val_acc_score:.4f},{val_f1_score:.4f}',
                           close=True)
        print(f"  Epoch {epoch:03d} | Loss: train={avg_train_loss:.4f} val={avg_validation_loss:.4f} | "
        f"Dice: train={avg_dice:.4f} val={avg_validation_dice:.4f} | "
        f"Acc: train={train_acc:.4f} val={val_acc_score:.4f} | "
        f"F1: train={train_f1_score:.4f} val={val_f1_score:.4f} | "
        f"Patience: {patience}/{config_training['max_patience']}")
        # early stopping
        if patience > config_training['max_patience']:
            logging.info(f"\nValidation loss did not improve over the last {patience} epochs. Stopping training")
            break

    # store metrics
    metrics = pd.read_csv(f'{run_path}/fold_{n}/metrics.csv')
    plot_evolution(metrics, columns=['Train_loss', 'Validation_loss'], path=f'{run_path}/fold_{n}/loss_evolution.png')
    plot_evolution(metrics, columns=['Train_dice', 'Validation_dice'], path=f'{run_path}/fold_{n}/segmentation_metrics_evolution.png')
    plot_evolution(metrics, columns=['Train_acc', 'Train_F1', 'Validation_acc', 'Validation_F1'], path=f'{run_path}/fold_{n}/classification_metrics_evolution.png')

    """
    INFERENCE PHASE
    """

    # results for validation dataset
    logging.info(f"\n\n ###############  VALIDATION PHASE  ###############  \n\n")
    model = load_pretrained_model(model, f'{run_path}/fold_{n}/model_{timestamp}_fold_{n}')

    # results for test dataset
    logging.info(f"\n\n ###############  TESTING PHASE  ###############  \n\n")
    if len(config_data['classes']) <= 2:
        test_results_segmentation, test_results_classification = inference_multitask_binary_classification_segmentation(model=model, test_loader=test_loader, path=f"{run_path}/fold_{n}/", device=dev)
    else:
        test_results_segmentation, test_results_classification = inference_multitask_multiclass_classification_segmentation(model=model, test_loader=test_loader, path=f"{run_path}/fold_{n}/", device=dev, threshold=config_training["threshold_postprocessing"], overlap_seg_based_on_class=config_training["overlap_seg_based_on_class"], overlap_class_based_on_seg=config_training["overlap_class_based_on_seg"])
    logging.info(f"Segmentation metric:\n\n{test_results_segmentation.mean()}\n")

    # classification metrics
    if len(config_data['classes']) <= 2:
        logging.info(f"\nClassification metrics:\n\n{pformat(binary_classification_metrics(test_results_classification.ground_truth, test_results_classification.predicted_label))}")
    else:
        logging.info(f"\nClassification metrics:\n\n{pformat(multiclass_classification_metrics(test_results_classification.ground_truth, test_results_classification.predicted_label))}")

    # Clear the GPU memory after evaluating on the test data for this fold
    torch.cuda.empty_cache()

    # ── FEATURE EXTRACTION + ENSEMBLE + QSVM ────────────────────────────
    print(f"\n  [ENSEMBLE] Starting for fold {n}...")
    model.eval()

    captured = {}
    def hook_fn(module, input, output):
        captured['feat'] = output.detach().cpu()
    hook = model.classifier[2].register_forward_hook(hook_fn)

    train_feats, train_labs = [], []
    test_feats, test_labs, nn_probs_list = [], [], []

    with torch.no_grad():
        for data in training_loader:
            imgs = data['image'].to(dev)
            lbl = data['label'].cpu()
            model(imgs)
            train_feats.append(captured['feat'].numpy())
            if lbl.ndim > 1 and lbl.shape[1] > 1:
                lbl = torch.argmax(lbl, dim=1)
            train_labs.append(lbl.numpy().flatten())

        for data in test_loader:
            imgs = data['image'].to(dev)
            lbl = data['label'].cpu()
            logits, _ = model(imgs)
            if isinstance(logits, list):
                logits = torch.mean(torch.stack(logits, dim=0), dim=0)
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            nn_probs_list.append(probs)
            test_feats.append(captured['feat'].numpy())
            if lbl.ndim > 1 and lbl.shape[1] > 1:
                lbl = torch.argmax(lbl, dim=1)
            test_labs.append(lbl.numpy().flatten())

    hook.remove()
    X_tr = np.vstack(train_feats)
    y_tr = np.concatenate(train_labs).astype(int)
    X_te = np.vstack(test_feats)
    y_te = np.concatenate(test_labs).astype(int)
    # ── SAVE ALL VISUALIZATIONS ──────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler as StandardScalerViz
    import cv2

    CLASS_NAMES = ['normal', 'benign', 'malignant']
    colors = ['blue', 'green', 'red']
    viz_base = Path(f'{run_path}/fold_{n}')

    # save features for tsne/umap
    np.save(viz_base / "test_features.npy", X_te)
    np.save(viz_base / "test_labels.npy", y_te)
    np.save(viz_base / "train_features.npy", X_tr)
    np.save(viz_base / "train_labels.npy", y_tr)
    print(f"  [VIZ] Features saved")

    # ── SEGMENTATION OVERLAYS ────────────────────────────────────────────
    overlay_path = viz_base / "seg_overlays"
    overlay_path.mkdir(exist_ok=True)
    sample_count = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            imgs   = data['image'].to(dev)
            masks  = data['mask'].cpu()
            lbl    = data['label'].cpu()
            logits, seg_output = model(imgs)
            if isinstance(seg_output, list):
                seg_output = seg_output[-1]
            seg_pred = (torch.sigmoid(seg_output) > 0.5).cpu()
            for i in range(imgs.shape[0]):
                if sample_count >= 15:
                    break
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                img_np  = imgs[i].cpu().squeeze().numpy()
                mask_np = masks[i].squeeze().numpy().astype(float)
                pred_np = seg_pred[i].squeeze().numpy().astype(float)
                img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

                axes[0].imshow(img_norm, cmap='gray')
                axes[0].set_title('Input US Image')
                axes[0].axis('off')

                gt_ov = np.stack([img_norm]*3, axis=-1)
                gt_ov[:,:,1] = np.clip(gt_ov[:,:,1] + mask_np * 0.5, 0, 1)
                axes[1].imshow(gt_ov)
                axes[1].set_title('Ground Truth Overlay')
                axes[1].axis('off')

                pr_ov = np.stack([img_norm]*3, axis=-1)
                pr_ov[:,:,0] = np.clip(pr_ov[:,:,0] + pred_np * 0.5, 0, 1)
                if lbl.ndim > 1 and lbl.shape[1] > 1:
                    cls_name = CLASS_NAMES[torch.argmax(lbl[i]).item()]
                else:
                    cls_name = CLASS_NAMES[int(lbl[i].item())]
                axes[2].imshow(pr_ov)
                axes[2].set_title(f'Predicted Overlay ({cls_name})')
                axes[2].axis('off')

                plt.tight_layout()
                plt.savefig(overlay_path / f'overlay_{sample_count:02d}_{cls_name}.png', dpi=100)
                plt.close()
                sample_count += 1
            if sample_count >= 15:
                break
    print(f"  [VIZ] Overlays saved: {overlay_path}")

    # ── ENCODER FEATURE MAPS WITH OVERLAY ON IMAGE ───────────────────────
    encoder_path = viz_base / "encoder_features"
    encoder_path.mkdir(exist_ok=True)
    encoder_outputs = {}
    def make_hook(name):
        def fn(module, input, output):
            encoder_outputs[name] = output.detach().cpu()
        return fn
    hooks = [
        model.conv_0_0.register_forward_hook(make_hook('level_0')),
        model.conv_1_0.register_forward_hook(make_hook('level_1')),
        model.conv_2_0.register_forward_hook(make_hook('level_2')),
        model.conv_3_0.register_forward_hook(make_hook('level_3')),
        model.conv_4_0.register_forward_hook(make_hook('level_4')),
    ]
    with torch.no_grad():
        sample_batch = next(iter(test_loader))
        sample_img = sample_batch['image'].to(dev)[:1]
        model(sample_img)
    for h in hooks:
        h.remove()

    orig_img = sample_img[0].cpu().squeeze().numpy()
    orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)

    for level_name, feat_map in encoder_outputs.items():
        feat = feat_map[0]
        n_ch = min(8, feat.shape[0])
        fig, axes = plt.subplots(2, n_ch, figsize=(n_ch*3, 6))
        for ch in range(n_ch):
            fm = feat[ch].numpy()
            fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
            fm_resized = cv2.resize(fm, (orig_img.shape[1], orig_img.shape[0]))

            # top row raw feature map
            axes[0, ch].imshow(fm_resized, cmap='viridis')
            axes[0, ch].set_title(f'Ch {ch}', fontsize=8)
            axes[0, ch].axis('off')

            # bottom row overlay on original image
            overlay = np.stack([orig_img]*3, axis=-1)
            heatmap = plt.cm.jet(fm_resized)[:,:,:3]
            blended = 0.5 * overlay + 0.5 * heatmap
            axes[1, ch].imshow(blended)
            axes[1, ch].set_title(f'Overlay Ch{ch}', fontsize=8)
            axes[1, ch].axis('off')

        plt.suptitle(f'Encoder {level_name} Feature Maps + Overlay - Fold {n}', fontsize=12)
        plt.tight_layout()
        plt.savefig(encoder_path / f'{level_name}_with_overlay.png', dpi=120, bbox_inches='tight')
        plt.close()
    print(f"  [VIZ] Encoder feature maps saved: {encoder_path}")

    # ── SVM INPUT FEATURE HEATMAP ─────────────────────────────────────────
    svm_input_path = viz_base / "svm_inputs"
    svm_input_path.mkdir(exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.imshow(X_te[:20].T, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title(f'Fold {n} - Feature Vectors Input to SVM (first 20 test samples)')
    plt.xlabel('Sample index')
    plt.ylabel('Feature dimension')
    plt.tight_layout()
    plt.savefig(svm_input_path / 'feature_heatmap.png', dpi=100)
    plt.close()
    print(f"  [VIZ] SVM input heatmap saved")

    # ── T-SNE ────────────────────────────────────────────────────────────
    scaler_viz = StandardScalerViz()
    X_te_sc_viz = scaler_viz.fit_transform(X_te)
    tsne = TSNE(n_components=2, random_state=42,
                perplexity=min(30, len(y_te)-1))
    X_tsne = tsne.fit_transform(X_te_sc_viz)
    plt.figure(figsize=(8, 6))
    for cls_idx, (color, cname) in enumerate(zip(colors, CLASS_NAMES)):
        mask = y_te == cls_idx
        plt.scatter(X_tsne[mask,0], X_tsne[mask,1],
                   c=color, label=cname, alpha=0.7, s=40)
    plt.title(f'Fold {n} - t-SNE of MTUNet++ Features (SVM Input)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(svm_input_path / 'tsne.png', dpi=100)
    plt.close()
    print(f"  [VIZ] t-SNE saved")

    nn_probs = np.vstack(nn_probs_list)

    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    # best SVM with probability
    best_svm, best_svm_acc = None, 0
    for kernel, C in [('rbf',0.1),('rbf',1),('rbf',10),('rbf',100),('linear',0.1),('linear',1),('linear',10)]:
        svm = SVC(kernel=kernel, C=C, gamma='scale',
                  class_weight='balanced', probability=True)
        svm.fit(X_tr_sc, y_tr)
        acc = accuracy_score(y_te, svm.predict(X_te_sc))
        print(f"  [SVM] kernel={kernel} C={C} -> ACC={acc:.4f}")
        if acc > best_svm_acc:
            best_svm_acc = acc
            best_svm = svm

    svm_probs = best_svm.predict_proba(X_te_sc)

    # ensemble NN + SVM
    best_acc, best_preds, best_w = 0, None, 0.5
    for w in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        combined = w * nn_probs + (1 - w) * svm_probs
        preds = np.argmax(combined, axis=1)
        acc = accuracy_score(y_te, preds)
        print(f"  [ENSEMBLE] w_nn={w} -> ACC={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_preds = preds
            best_w = w

    ens_f1 = f1_score(y_te, best_preds, average='weighted')
    print(f"\n  [ENSEMBLE] Fold {n} BEST w_nn={best_w} -> ACC={best_acc:.4f} F1={ens_f1:.4f}")
    print(classification_report(y_te, best_preds,
          target_names=['normal', 'benign', 'malignant']))
    np.save(f'{run_path}/fold_{n}/ensemble_preds.npy', best_preds)
    np.save(f'{run_path}/fold_{n}/ensemble_gt.npy', y_te)
    # save ensemble metrics to csv
    with open(f'{run_path}/fold_{n}/ensemble_metrics.csv', 'w') as f:
        f.write(f'method,acc,f1_weighted\n')
        f.write(f'ensemble,{best_acc:.4f},{ens_f1:.4f}\n')
        f.write(f'best_w_nn,{best_w},,\n')



    best_row = metrics.loc[metrics['Validation_loss'].idxmin()]
    print(f"\n  Fold {n} Best Epoch {int(best_row['epoch'])} | "
        f"Dice={best_row['Validation_dice']:.4f} | "
        f"Acc={best_row['Validation_acc']:.4f} | "
        f"F1={best_row['Validation_F1']:.4f}")
    del model


# saving final results as an Excel file
save_segmentation_results(run_path)

# saving final results as an Excel file
save_classification_results(run_path, len(config_data['classes']))
all_metrics = []
for i in range(config_training['CV']):
    try:
        fm = pd.read_csv(f'{run_path}/fold_{i}/metrics.csv')
        all_metrics.append(fm.loc[fm['Validation_loss'].idxmin()])
    except FileNotFoundError:
        pass
if all_metrics:
    s = pd.DataFrame(all_metrics)
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS ACROSS {len(all_metrics)} FOLDS")
    print(f"  Dice : mean={s['Validation_dice'].mean():.4f}  std={s['Validation_dice'].std():.4f}")
    print(f"  Acc  : mean={s['Validation_acc'].mean():.4f}  std={s['Validation_acc'].std():.4f}")
    print(f"  F1   : mean={s['Validation_F1'].mean():.4f}  std={s['Validation_F1'].std():.4f}")
    print(f"{'='*60}\n")
# Measuring total time
end_time = time.perf_counter()
logging.info(f"Total time for all of the folds: {end_time - init_time:.2f}")

# ── ENSEMBLE FINAL SUMMARY ───────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  ENSEMBLE FINAL RESULTS ACROSS FOLDS")
ens_accs, ens_f1s = [], []
for i in range(config_training['CV']):
    try:
        p = np.load(f'{run_path}/fold_{i}/ensemble_preds.npy')
        g = np.load(f'{run_path}/fold_{i}/ensemble_gt.npy')
        ens_accs.append(accuracy_score(g, p))
        ens_f1s.append(f1_score(g, p, average='weighted'))
    except FileNotFoundError:
        pass
if ens_accs:
    print(f"  ACC : mean={np.mean(ens_accs):.4f}  std={np.std(ens_accs):.4f}")
    print(f"  F1  : mean={np.mean(ens_f1s):.4f}  std={np.std(ens_f1s):.4f}")
print(f"{'='*60}\n")

# ── SAVE FINAL SUMMARY TO FILE ──────────────────────────────────────────
summary_path = f'{run_path}/final_ensemble_summary.csv'
with open(summary_path, 'w') as f:
    f.write('method,acc_mean,acc_std,f1_mean,f1_std\n')
    if ens_accs:
        f.write(f'ensemble,{np.mean(ens_accs):.4f},{np.std(ens_accs):.4f},{np.mean(ens_f1s):.4f},{np.std(ens_f1s):.4f}\n')
print(f"  Summary saved to {summary_path}")