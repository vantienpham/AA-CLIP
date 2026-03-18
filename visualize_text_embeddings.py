import argparse
import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from dataset.constants import CLASS_NAMES, PROMPTS, REAL_NAMES
from model.adapter import AdaptedCLIP
from model.clip import create_model
from model.tokenizer import tokenize
from utils import setup_seed

warnings.filterwarnings("ignore")

cpu_num = 4

os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

COLORS = {0: "#2196F3", 1: "#F44336"}
LABEL_MAP = {0: "Normal", 1: "Abnormal"}
CENTROID_COLORS = {0: "#0D47A1", 1: "#B71C1C"}


def build_sentences(real_name: str):
    prompt_normal = PROMPTS["prompt_normal"]
    prompt_abnormal = PROMPTS["prompt_abnormal"]
    prompt_templates = PROMPTS["prompt_templates"]

    def expand(state_list):
        sentences = []
        for state in state_list:
            state_prompt = state.format(real_name)
            for template in prompt_templates:
                sentences.append(template.format(state_prompt))
        return sentences

    return expand(prompt_normal), expand(prompt_abnormal)


def centroid_distances(embeddings: np.ndarray, labels: np.ndarray):
    normal_center = embeddings[labels == 0].mean(axis=0)
    abnormal_center = embeddings[labels == 1].mean(axis=0)
    euclidean_distance = np.linalg.norm(normal_center - abnormal_center)
    cosine_distance = 1.0 - (normal_center @ abnormal_center) / (
        np.linalg.norm(normal_center) * np.linalg.norm(abnormal_center)
    )
    return euclidean_distance, cosine_distance


def draw_centroid(ax, coords_2d: np.ndarray, labels: np.ndarray, label_index: int):
    class_mask = labels == label_index
    center_x = coords_2d[class_mask, 0].mean()
    center_y = coords_2d[class_mask, 1].mean()
    ax.scatter(
        center_x,
        center_y,
        c=CENTROID_COLORS[label_index],
        marker="*",
        s=350,
        edgecolors="black",
        linewidths=0.6,
        zorder=5,
        label=f"{LABEL_MAP[label_index]} centroid",
    )
    return center_x, center_y


def plot_normal_abnormal(
    coords_2d: np.ndarray,
    all_embeddings: np.ndarray,
    all_labels: np.ndarray,
    base_title: str,
    save_path: str,
):
    euclidean_distance, cosine_distance = centroid_distances(all_embeddings, all_labels)
    title = (
        f"{base_title}\nCentroid dist (768-d) — "
        f"Euclidean: {euclidean_distance:.4f}  |  Cosine: {cosine_distance:.4f}"
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    for label_index in [0, 1]:
        class_mask = all_labels == label_index
        ax.scatter(
            coords_2d[class_mask, 0],
            coords_2d[class_mask, 1],
            c=COLORS[label_index],
            label=LABEL_MAP[label_index],
            s=22,
            alpha=0.7,
            edgecolors="none",
        )

    center_normal_x, center_normal_y = draw_centroid(ax, coords_2d, all_labels, 0)
    center_abnormal_x, center_abnormal_y = draw_centroid(ax, coords_2d, all_labels, 1)
    ax.plot(
        [center_normal_x, center_abnormal_x],
        [center_normal_y, center_abnormal_y],
        color="gray",
        linestyle="--",
        linewidth=1.2,
        zorder=4,
    )

    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_per_class(
    coords_2d: np.ndarray,
    all_embeddings: np.ndarray,
    all_labels: np.ndarray,
    all_class_names: np.ndarray,
    base_title: str,
    save_path: str,
):
    unique_classes = sorted(set(all_class_names))
    ncols = 5
    nrows = int(np.ceil(len(unique_classes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for axis_index, cls in enumerate(unique_classes):
        ax = axes[axis_index]
        class_mask = all_class_names == cls
        class_embeddings = all_embeddings[class_mask]
        class_labels = all_labels[class_mask]
        class_coords_2d = coords_2d[class_mask]

        for label_index in [0, 1]:
            label_mask = class_labels == label_index
            ax.scatter(
                class_coords_2d[label_mask, 0],
                class_coords_2d[label_mask, 1],
                c=COLORS[label_index],
                label=LABEL_MAP[label_index],
                s=35,
                alpha=0.9,
                edgecolors="none",
            )

        center_normal_x, center_normal_y = draw_centroid(
            ax, class_coords_2d, class_labels, 0
        )
        center_abnormal_x, center_abnormal_y = draw_centroid(
            ax, class_coords_2d, class_labels, 1
        )
        ax.plot(
            [center_normal_x, center_abnormal_x],
            [center_normal_y, center_abnormal_y],
            color="gray",
            linestyle="--",
            linewidth=1.0,
            zorder=4,
        )

        euclidean_distance, cosine_distance = centroid_distances(
            class_embeddings, class_labels
        )
        short_name = cls.split("/")[-1]
        ax.set_title(
            f"{short_name}\nEuc: {euclidean_distance:.3f} | Cos: {cosine_distance:.3f}",
            fontsize=9,
        )
        ax.legend(fontsize=6)

    for ax in axes[len(unique_classes) :]:
        ax.set_visible(False)

    fig.suptitle(base_title, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize text embeddings")
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-L-14-336",
        help="ViT-L-14-336",
    )
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--relu", action="store_true")
    parser.add_argument("--datasets", type=str, nargs="+", default=["MVTec"])
    parser.add_argument(
        "--ckpt_path", type=str, default="ckpt/baseline/text_adapter.pth"
    )
    parser.add_argument("--save_path", type=str, default="ckpt/baseline")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--text_adapt_weight", type=float, default=0.1)
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_until", type=int, default=3)
    parser.add_argument("--image_adapt_until", type=int, default=6)

    args = parser.parse_args()
    # ========================================================
    setup_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.save_path, "visualize_text_embeddings.log"),
        encoding="utf-8",
        level=logging.INFO,
    )
    logger.info("args: %s", vars(args))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    logger.info("using device: %s", device)

    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()

    model = AdaptedCLIP(
        clip_model=clip_model,
        text_adapt_weight=args.text_adapt_weight,
        image_adapt_weight=args.image_adapt_weight,
        text_adapt_until=args.text_adapt_until,
        image_adapt_until=args.image_adapt_until,
        relu=args.relu,
    ).to(device)
    model.eval()

    text_ckpt = os.path.join(args.save_path, "text_adapter.pth")
    if not os.path.isfile(text_ckpt):
        raise FileNotFoundError(f"Missing {text_ckpt}")
    logger.info("loading text adapter from %s", text_ckpt)
    text_ckpt = torch.load(text_ckpt, map_location=device)
    model.text_adapter.load_state_dict(text_ckpt["text_adapter"])

    all_embeddings = []
    all_labels = []
    all_class_names = []

    with torch.no_grad():
        for dataset_name in args.datasets:
            if dataset_name not in CLASS_NAMES:
                logger.warning("skip unsupported dataset: %s", dataset_name)
                continue

            for class_name in CLASS_NAMES[dataset_name]:
                real_name = REAL_NAMES[dataset_name][class_name]
                normal_sentences, abnormal_sentences = build_sentences(real_name)
                for label_index, sentences in enumerate(
                    [normal_sentences, abnormal_sentences]
                ):
                    tokens = tokenize(sentences).to(device)
                    embeddings = model.encode_text(tokens)
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                    for embedding in embeddings.cpu().numpy():
                        all_embeddings.append(embedding)
                        all_labels.append(label_index)
                        all_class_names.append(f"{dataset_name}/{class_name}")

    if len(all_embeddings) == 0:
        raise RuntimeError("No text embeddings collected. Check --datasets input.")

    all_embeddings = np.stack(all_embeddings)
    all_labels = np.array(all_labels)
    all_class_names = np.array(all_class_names)

    logger.info(
        "total embeddings: %d (normal=%d, abnormal=%d)",
        len(all_embeddings),
        int((all_labels == 0).sum()),
        int((all_labels == 1).sum()),
    )

    global_euclidean_distance, global_cosine_distance = centroid_distances(
        all_embeddings, all_labels
    )
    logger.info(
        "global centroid distances - euclidean: %.4f, cosine: %.4f",
        global_euclidean_distance,
        global_cosine_distance,
    )

    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(all_embeddings)
    explained_variance = pca.explained_variance_ratio_
    logger.info(
        "PCA explained variance - PC1: %.3f, PC2: %.3f",
        explained_variance[0],
        explained_variance[1],
    )

    tsne = TSNE(n_components=2, perplexity=30, verbose=1)
    tsne_2d = tsne.fit_transform(all_embeddings)

    datasets_title = ",".join(args.datasets)

    pca_normal_path = os.path.join(args.save_path, "pca_normal_vs_abnormal.png")
    pca_per_class_path = os.path.join(args.save_path, "pca_per_class.png")
    tsne_normal_path = os.path.join(args.save_path, "tsne_normal_vs_abnormal.png")
    tsne_per_class_path = os.path.join(args.save_path, "tsne_per_class.png")

    plot_normal_abnormal(
        coords_2d=pca_2d,
        all_embeddings=all_embeddings,
        all_labels=all_labels,
        base_title=f"PCA - text embeddings (stage-1 projector) | {datasets_title} - all classes",
        save_path=pca_normal_path,
    )
    plot_per_class(
        coords_2d=pca_2d,
        all_embeddings=all_embeddings,
        all_labels=all_labels,
        all_class_names=all_class_names,
        base_title=f"PCA - text embeddings per class ({datasets_title})",
        save_path=pca_per_class_path,
    )

    plot_normal_abnormal(
        coords_2d=tsne_2d,
        all_embeddings=all_embeddings,
        all_labels=all_labels,
        base_title=f"t-SNE - text embeddings (stage-1 projector) | {datasets_title} - all classes",
        save_path=tsne_normal_path,
    )
    plot_per_class(
        coords_2d=tsne_2d,
        all_embeddings=all_embeddings,
        all_labels=all_labels,
        all_class_names=all_class_names,
        base_title=f"t-SNE - text embeddings per class ({datasets_title})",
        save_path=tsne_per_class_path,
    )

    logger.info("saved plots: %s", pca_normal_path)
    logger.info("saved plots: %s", pca_per_class_path)
    logger.info("saved plots: %s", tsne_normal_path)
    logger.info("saved plots: %s", tsne_per_class_path)
    logging.shutdown()


if __name__ == "__main__":
    main()
