#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export two publication-ready figures from a real NDPNet/NDCN checkpoint:

1) mask_visualization.{pdf,png}
   Input image | NDM assignment maps | component scores | final soft mask |
   mask overlay.  The support/query masks are captured *inside the real model
   forward*, so alpha_norm='supminmax' uses the same support-set range as the
   classifier did.

2) tsne_uniform_vs_masked.{pdf,png}
   Joint t-SNE of uniform-pooled and mask-pooled query vectors.  Both vectors
   are computed from the same backbone output and the same query images.  A
   single 5-way episode is used because episodic labels 0..4 are remapped for
   every task and therefore must not be mixed across unrelated episodes.

The script also saves visual_data.npz and visual_metadata.json, which preserve
the numerical evidence used by the figures.

Place this file in the DN4/NDPNet project root (next to Test_NDCN.py), then run:

    python Export_NDPNet_Visuals.py \
      --dataset-dir /data/Datasets/CUB_birds/For_FewShot \
      --checkpoint ./results/YOUR_RUN/model_best.pth.tar \
      --encoder-model ResNet12_ReLU \
      --shot-num 5 \
      --output-dir ./paper_visuals

If --checkpoint points to a run directory, model_best.pth.tar is appended.
Dependencies beyond the training repository: matplotlib and scikit-learn.
"""

from __future__ import print_function

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize
except ImportError as exc:
    raise SystemExit(
        "Missing plotting dependency. Install it with: "
        "pip install matplotlib scikit-learn"
    ) from exc

sys.dont_write_bytecode = True

import dataset.general_dataloader as FewShotDataloader
import models.network as FewShotNet
from models.ndcn import make_nonneg


PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
           "#56B4E9", "#000000", "#F0E442", "#7A3E9D", "#8C564B"]


def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Export real-checkpoint NDPNet mask and t-SNE figures.")
    parser.add_argument("--dataset-dir", required=True,
                        help="Dataset root containing images/ and test.csv.")
    parser.add_argument("--checkpoint", required=True,
                        help="model_best.pth.tar or its containing run directory.")
    parser.add_argument("--output-dir", default="./paper_visuals")
    parser.add_argument("--data-name", default="CubBird")
    parser.add_argument("--encoder-model", default="ResNet12_ReLU")
    parser.add_argument("--classifier-model", default="NDCN")
    parser.add_argument("--way-num", type=int, default=5)
    parser.add_argument("--shot-num", type=int, default=5)
    parser.add_argument("--query-num", type=int, default=15)
    parser.add_argument("--image-size", type=int, default=84)
    parser.add_argument("--episode-index", type=int, default=0,
                        help="Deterministic test episode to visualize; do not tune it on accuracy.")
    parser.add_argument("--mask-vmax", default=None,
                        help="Mask colour scale. A number fixes it (1.0 = absolute). "
                             "'row' scales each row to its own maximum, which reveals "
                             "structure in rows whose occupancy is very low. Default: "
                             "the largest value across the shown rows.")
    parser.add_argument("--samples-per-class", type=int, default=1,
                        help="Rows per episode class in the mask figure.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--tsne-perplexity", type=float, default=20.0)
    parser.add_argument("--tsne-iterations", type=int, default=1500)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--cpu", action="store_true")
    return parser


def complete_dataloader_options(args):
    """Add the option names expected by the original DN4 dataloader."""
    args.dataset_dir = args.dataset_dir
    args.data_name = args.data_name
    args.imageSize = args.image_size
    args.mode = "test"
    args.train_aug = False
    args.test_aug = False
    args.episodeSize = 1
    args.testepisodeSize = 1
    args.epochs = 1
    args.episode_train_num = 1
    args.episode_val_num = 1
    args.episode_test_num = args.episode_index + 1
    args.aug_shot_num = 20
    args.neighbor_k = 3
    args.cuda = torch.cuda.is_available() and not args.cpu
    args.ngpu = 1
    return args


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def resolve_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "model_best.pth.tar"
    if not path.is_file():
        raise FileNotFoundError("Checkpoint not found: %s" % path)
    return path


def load_checkpoint(path, device):
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        return torch.load(str(path), map_location=device)


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def cpu_copy(value):
    if value is None:
        return None
    return value.detach().float().cpu().clone()


def capture_real_masks(model, input_q, input_s):
    """Run one real forward while tracing the two internal _mask calls."""
    core = unwrap(model)
    classifier = core.classifier
    if not hasattr(classifier, "_mask") or not hasattr(classifier, "_pool"):
        raise RuntimeError("The selected classifier does not expose NDPNet masks.")
    if getattr(classifier, "uniform_mask", False):
        raise RuntimeError(
            "This checkpoint/config has uniform_mask=True, so it cannot produce "
            "NDM assignment maps. Use the checkpoint trained with the mask.")
    if getattr(classifier, "frontend", None) != "ndm":
        raise RuntimeError(
            "Assignment-map visualization requires frontend='ndm'; got %r."
            % getattr(classifier, "frontend", None))

    calls = []
    original_mask = classifier._mask

    # This function is assigned to the instance, not the class, so Python calls
    # it with exactly (Z, ref); classifier is captured by the closure.
    def traced_mask(Z, ref):
        range_before = getattr(classifier, "_alpha_range", None)
        M, alpha, A = original_mask(Z, ref)
        raw = getattr(classifier, "_raw_s", None)
        calls.append({
            "Z": cpu_copy(Z),
            "ref": cpu_copy(ref),
            "mask": cpu_copy(M),
            "alpha": cpu_copy(alpha),
            "assignments": cpu_copy(A),
            "raw_scores": cpu_copy(raw),
            "range_before": None if range_before is None else
                [float(range_before[0]), float(range_before[1])],
        })
        return M, alpha, A

    classifier._mask = traced_mask
    try:
        # is_feature=True returns the exact feature maps used by the classifier;
        # it does not run the backbone a second time.
        feat_q, feat_s, logits = model(input_q, input_s, is_feature=True)
    finally:
        classifier._mask = original_mask

    if len(calls) != 2:
        raise RuntimeError(
            "Expected two mask calls (support, query), captured %d." % len(calls))
    support_call, query_call = calls
    if query_call["assignments"] is None or query_call["alpha"] is None:
        raise RuntimeError("The real query forward did not return NDM assignments/alpha.")

    # Recreate the exact local-descriptor tensor from the returned feature map.
    Zq = make_nonneg(classifier._flatten(feat_q),
                     classifier.nonneg_mode, classifier.eps)
    Mq = query_call["mask"].to(device=Zq.device, dtype=Zq.dtype)
    masked, _ = classifier._pool(Zq, Mq, classifier.eps,
                                 classifier.pool_floor)
    uniform, _ = classifier._pool(Zq, torch.ones_like(Mq), classifier.eps,
                                  classifier.pool_floor)

    return {
        "logits": logits.detach().float().cpu(),
        "feat_q": feat_q.detach().float().cpu(),
        "feat_s": feat_s.detach().float().cpu(),
        "support": support_call,
        "query": query_call,
        "masked_vectors": masked.detach().float().cpu(),
        "uniform_vectors": uniform.detach().float().cpu(),
    }


def prepare_episode(batch, device):
    query_images, query_targets, support_images, _ = batch
    input_q = torch.cat(query_images, 0).to(device, non_blocking=True)
    input_s = torch.cat(support_images, 0).squeeze(0).to(device, non_blocking=True)
    input_s = input_s.contiguous().view(
        -1, input_s.size(2), input_s.size(3), input_s.size(4))
    target = torch.cat(query_targets, 0).long().to(device, non_blocking=True)
    return input_q, input_s, target


def denormalize_images(tensor):
    # The provided DN4 dataloader uses Normalize((.5,.5,.5), (.5,.5,.5)).
    return ((tensor.detach().cpu() * 0.5 + 0.5).clamp(0, 1)
            .permute(0, 2, 3, 1).numpy())


def infer_map_shape(length):
    side = int(round(math.sqrt(length)))
    if side * side != length:
        raise RuntimeError(
            "Local-descriptor count L=%d is not square; provide a custom reshape."
            % length)
    return side, side


def class_lookup(dataset_dir):
    lookup = {}
    csv_path = Path(dataset_dir) / "test.csv"
    if not csv_path.is_file():
        return lookup
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                lookup[os.path.basename(row[0])] = row[1]
    return lookup


def episode_query_paths(loader, episode_index):
    episode = loader.dataset.data_list[episode_index]
    return [path for cls in episode for path in cls["query_img"]]


def select_stratified(labels, samples_per_class):
    chosen = []
    for label in sorted(np.unique(labels).tolist()):
        indices = np.flatnonzero(labels == label)
        chosen.extend(indices[:samples_per_class].tolist())
    return np.asarray(chosen, dtype=np.int64)


def parse_mask_scale(value):
    """CLI accepts a number, the word 'row', or nothing."""
    if value is None:
        return None
    if str(value).lower() == "row":
        return "row"
    return float(value)


def save_mask_figure(images, labels, predictions, class_names, masks,
                     assignments, alpha, samples_per_class, output_dir, dpi,
                     mask_vmax=None):
    """
    One row per query image: input, the r assignment maps, the soft mask, overlay.

    Two things differ from a naive layout.

    Components are re-ordered per row by descending alpha, and the columns are
    labelled by rank rather than by index. NMF component indices carry no
    identity across images -- component 1 of one image and component 1 of
    another are unrelated -- so a fixed column per index would imply a
    correspondence that does not exist. Ranking is well defined per image.

    Alpha is printed inside every panel instead of once in the header, because
    each image has its own scores.
    """
    n, length, rank = assignments.shape
    height, width = infer_map_shape(length)
    chosen = select_stratified(labels, max(1, samples_per_class))
    nrows = len(chosen)
    ncols = rank + 3  # input + rank assignments + final mask + overlay

    # The mask occupies a narrow band well below 1, so a fixed vmax=1 renders
    # every panel nearly black. Scale to the values actually present, and state
    # the scale in the axis label so the figure stays readable as evidence.
    per_row = (str(mask_vmax).lower() == "row")
    if per_row:
        mask_vmax = None
    elif mask_vmax is None:
        mask_vmax = float(max(masks[chosen].max(), 1e-6))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(1.55 * ncols, 1.62 * nrows),
        squeeze=False, constrained_layout=True)

    for row, index in enumerate(chosen):
        image = images[index]
        axes[row, 0].imshow(image, interpolation="nearest", rasterized=True)
        true_label = int(labels[index])
        pred_label = int(predictions[index])
        name = class_names[index] if class_names[index] else "class %d" % true_label
        status = "correct" if true_label == pred_label else "wrong"
        axes[row, 0].set_ylabel(
            "%s\nGT %d / Pred %d (%s)" % (name, true_label, pred_label, status),
            fontsize=7.5)

        order = np.argsort(-alpha[index])          # per-image, strongest first
        for slot, component in enumerate(order):
            component_map = assignments[index, :, component].reshape(height, width)
            ax = axes[row, slot + 1]
            ax.imshow(component_map, cmap="viridis", vmin=0.0, vmax=1.0,
                      interpolation="nearest", rasterized=True)
            ax.text(0.03, 0.97, r"$\alpha=%.2f$" % alpha[index, component],
                    transform=ax.transAxes, fontsize=7, color="white",
                    va="top", ha="left",
                    bbox=dict(facecolor="black", alpha=0.45, pad=1.0,
                              edgecolor="none"))
            if row == 0:
                ax.set_title("component\nranked %d" % (slot + 1), fontsize=8)

        mask_map = masks[index].reshape(height, width)
        # A shared scale renders a row whose mask sits near 0.04 as solid
        # black, hiding whatever structure it has. Per-row scaling shows the
        # structure; the absolute range is printed so the reader still sees
        # how low the occupancy is.
        row_vmax = float(mask_map.max()) if per_row else mask_vmax
        row_vmax = max(row_vmax, 1e-6)
        axes[row, rank + 1].imshow(
            mask_map, cmap="inferno", vmin=0.0, vmax=row_vmax,
            interpolation="nearest", rasterized=True)
        note = (r"$\bar{M}=%.2f$" % float(mask_map.mean())) if not per_row else \
               (r"$\bar{M}=%.2f$, max $%.2f$" % (float(mask_map.mean()), row_vmax))
        axes[row, rank + 1].text(
            0.03, 0.97, note,
            transform=axes[row, rank + 1].transAxes, fontsize=7, color="white",
            va="top", ha="left",
            bbox=dict(facecolor="black", alpha=0.45, pad=1.0, edgecolor="none"))

        upsampled = F.interpolate(
            torch.from_numpy(mask_map)[None, None].float(),
            size=image.shape[:2], mode="bilinear", align_corners=False
        )[0, 0].numpy()
        # Modulate the image brightness by the mask instead of blending a
        # colormap over it. A mostly dark colormap at alpha=0.48 washes the
        # photograph out; keeping the masked regions at full brightness and
        # dimming the rest shows what the pooling actually keeps.
        w = upsampled / max(upsampled.max(), 1e-6)
        dimmed = image * (0.20 + 0.80 * w[..., None])
        axes[row, rank + 2].imshow(np.clip(dimmed, 0.0, 1.0),
                                   interpolation="bilinear", rasterized=True)

        for col in range(ncols):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    axes[0, 0].set_title("Query image", fontsize=9)
    axes[0, rank + 1].set_title(
        "Soft mask $M$\n" + ("(scaled per row)" if per_row
                             else "(scaled to %.2f)" % mask_vmax), fontsize=8)
    axes[0, rank + 2].set_title("Masked image", fontsize=9)

    pdf_path = output_dir / "mask_visualization.pdf"
    png_path = output_dir / "mask_visualization.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return chosen, pdf_path, png_path


def make_joint_tsne(uniform_vectors, masked_vectors, labels, args):
    uniform_vectors = normalize(uniform_vectors, norm="l2")
    masked_vectors = normalize(masked_vectors, norm="l2")
    joint = np.concatenate([uniform_vectors, masked_vectors], axis=0)
    n_samples, dim = joint.shape
    pca_dim = min(50, dim, n_samples - 1)
    if pca_dim >= 2:
        joint = PCA(n_components=pca_dim, random_state=args.seed).fit_transform(joint)

    perplexity = min(float(args.tsne_perplexity), max(2.0, n_samples - 1.0))
    common = dict(
        n_components=2, perplexity=perplexity, learning_rate="auto",
        init="pca", random_state=args.seed, method="barnes_hut")
    try:
        coords = TSNE(max_iter=args.tsne_iterations, **common).fit_transform(joint)
    except TypeError:  # older scikit-learn calls this option n_iter
        coords = TSNE(n_iter=args.tsne_iterations, **common).fit_transform(joint)

    count = len(labels)
    return coords[:count], coords[count:], uniform_vectors, masked_vectors


def save_tsne_figure(uniform_xy, masked_xy, labels, uniform_vectors,
                     masked_vectors, output_dir, dpi):
    labels = np.asarray(labels)
    classes = sorted(np.unique(labels).tolist())
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.95), constrained_layout=True)

    all_xy = np.concatenate([uniform_xy, masked_xy], axis=0)
    xpad = max(np.ptp(all_xy[:, 0]) * 0.06, 1e-6)
    ypad = max(np.ptp(all_xy[:, 1]) * 0.06, 1e-6)
    xlim = (all_xy[:, 0].min() - xpad, all_xy[:, 0].max() + xpad)
    ylim = (all_xy[:, 1].min() - ypad, all_xy[:, 1].max() + ypad)

    # Put the silhouette on the panels. The two projections look alike by
    # construction, since a single t-SNE is fitted to both sets so that the
    # coordinates are comparable; without a number the reader has no way to
    # tell the panels apart.
    try:
        from sklearn.metrics import silhouette_score
        sil_u = silhouette_score(uniform_vectors, labels, metric="cosine")
        sil_m = silhouette_score(masked_vectors, labels, metric="cosine")
        titles = ["Uniform pooling\nsilhouette %.3f" % sil_u,
                  "NDPNet mask pooling\nsilhouette %.3f" % sil_m]
    except Exception:
        titles = ["Uniform pooling", "NDPNet mask pooling"]

    for axis, coords, title in zip(axes, [uniform_xy, masked_xy], titles):
        for class_id in classes:
            selected = labels == class_id
            axis.scatter(
                coords[selected, 0], coords[selected, 1], s=25,
                color=PALETTE[class_id % len(PALETTE)], marker="o",
                edgecolors="white", linewidths=0.35, alpha=0.9,
                rasterized=True)
        axis.set_title(title, fontsize=11)
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#777777")
            spine.set_linewidth(0.7)

    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=6,
                      markerfacecolor=PALETTE[c % len(PALETTE)],
                      markeredgecolor="white", label="Class %d" % c)
               for c in classes]
    fig.legend(handles=handles, loc="lower center", ncol=len(classes),
               bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=8)

    pdf_path = output_dir / "tsne_uniform_vs_masked.pdf"
    png_path = output_dir / "tsne_uniform_vs_masked.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # Report class separation on the full vectors; this is more meaningful than
    # a score computed only after the nonlinear 2-D projection.
    uniform_sil = float(silhouette_score(uniform_vectors, labels, metric="cosine"))
    masked_sil = float(silhouette_score(masked_vectors, labels, metric="cosine"))
    return uniform_sil, masked_sil, pdf_path, png_path


def json_safe_cfg(cfg):
    result = {}
    for key, value in dict(cfg or {}).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        else:
            result[key] = str(value)
    return result


def main(args):
    args = complete_dataloader_options(args)
    set_deterministic(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(checkpoint_path, device)
    ndcn_cfg = checkpoint.get("ndcn_cfg")
    if ndcn_cfg is None:
        raise RuntimeError(
            "Checkpoint has no ndcn_cfg. Use a checkpoint written by the revised "
            "Train_NDCN.py so the inference configuration is unambiguous.")

    model = FewShotNet.define_model(
        encoder_model=args.encoder_model,
        classifier_model=args.classifier_model,
        norm="batch", way_num=args.way_num, shot_num=args.shot_num,
        init_type="normal", use_gpu=args.cuda, ndcn_cfg=ndcn_cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Construct only test episodes.  episode_index is deterministic under seed.
    test_loader, = FewShotDataloader.get_Fewshot_dataloader(args, ["test"])
    selected_batch = None
    for index, batch in enumerate(test_loader):
        if index == args.episode_index:
            selected_batch = batch
            break
    if selected_batch is None:
        raise RuntimeError("Could not load episode index %d." % args.episode_index)

    input_q, input_s, target = prepare_episode(selected_batch, device)
    with torch.no_grad():
        trace = capture_real_masks(model, input_q, input_s)

    labels = target.detach().cpu().numpy()
    logits = trace["logits"].numpy()
    predictions = logits.argmax(axis=1)
    images = denormalize_images(input_q)
    masks = trace["query"]["mask"].numpy()
    alpha = trace["query"]["alpha"].numpy()
    assignments = trace["query"]["assignments"].numpy()
    uniform_vectors = trace["uniform_vectors"].numpy()
    masked_vectors = trace["masked_vectors"].numpy()

    query_paths = episode_query_paths(test_loader, args.episode_index)
    if len(query_paths) != len(labels):
        query_paths = [""] * len(labels)
    lookup = class_lookup(args.dataset_dir)
    class_names = [lookup.get(os.path.basename(path), "") for path in query_paths]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })

    chosen, mask_pdf, mask_png = save_mask_figure(
        images, labels, predictions, class_names, masks, assignments, alpha,
        args.samples_per_class, output_dir, args.dpi,
        mask_vmax=parse_mask_scale(args.mask_vmax))

    uniform_xy, masked_xy, uniform_norm, masked_norm = make_joint_tsne(
        uniform_vectors, masked_vectors, labels, args)
    uniform_sil, masked_sil, tsne_pdf, tsne_png = save_tsne_figure(
        uniform_xy, masked_xy, labels, uniform_norm, masked_norm,
        output_dir, args.dpi)

    np.savez_compressed(
        output_dir / "visual_data.npz",
        images_uint8=np.rint(images * 255.0).astype(np.uint8),
        labels=labels,
        predictions=predictions,
        logits=logits,
        query_masks=masks,
        query_alpha=alpha,
        query_assignments=assignments,
        uniform_vectors=uniform_vectors,
        masked_vectors=masked_vectors,
        uniform_tsne=uniform_xy,
        masked_tsne=masked_xy,
        selected_mask_indices=chosen,
        query_paths=np.asarray(query_paths),
        class_names=np.asarray(class_names),
    )

    accuracy = float((predictions == labels).mean() * 100.0)
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch_index"),
        "checkpoint_best_prec1": checkpoint.get("best_prec1"),
        "ndcn_cfg": json_safe_cfg(ndcn_cfg),
        "dataset_dir": str(Path(args.dataset_dir).expanduser().resolve()),
        "episode_index": args.episode_index,
        "seed": args.seed,
        "way_num": args.way_num,
        "shot_num": args.shot_num,
        "query_num": args.query_num,
        "episode_accuracy_percent": accuracy,
        "uniform_silhouette_cosine": uniform_sil,
        "masked_silhouette_cosine": masked_sil,
        "support_score_range_used_by_queries": trace["query"]["range_before"],
        "mask_mean": float(masks.mean()),
        "output_files": [p.name for p in
                         [mask_pdf, mask_png, tsne_pdf, tsne_png]],
        "note": (
            "Both t-SNE panels use the same query images and backbone features. "
            "The embeddings are jointly projected into one coordinate system. "
            "Only one episode is used because episodic labels are remapped per task."),
    }
    with (output_dir / "visual_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print("\nDone. Real-checkpoint outputs:")
    print("  %s" % mask_pdf)
    print("  %s" % mask_png)
    print("  %s" % tsne_pdf)
    print("  %s" % tsne_png)
    print("  %s" % (output_dir / "visual_data.npz"))
    print("  %s" % (output_dir / "visual_metadata.json"))
    print("Episode accuracy: %.2f%%" % accuracy)
    print("Silhouette (cosine, full vectors): uniform %.4f | masked %.4f" %
          (uniform_sil, masked_sil))
    if masked_sil <= uniform_sil:
        print("WARNING: this episode does not show improved class separation; "
              "report the result honestly and do not select another episode by accuracy.")


if __name__ == "__main__":
    ARGS = build_parser().parse_args()
    main(ARGS)
