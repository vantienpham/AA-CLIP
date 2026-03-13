import os
import argparse
import numpy as np
from tqdm import tqdm
import logging
from pandas import DataFrame, Series
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from utils import setup_seed
from model.adapter import AdaptedCLIP
from model.clip import create_model
from dataset import get_dataset, DOMAINS
from forward_utils import (
    get_adapted_text_embedding,
    calculate_similarity_map,
    metrics_eval,
    visualize,
)
import warnings

warnings.filterwarnings("ignore")

cpu_num = 4

os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_support_features(model, support_loader, device):
    all_features = []
    for (
        input_data
    ) in (
        support_loader
    ):  # bs always=1. training for an epoch first, Then use this updated model for memory bank construction.
        image = input_data[0].to(device)
        patch_tokens = model(image)
        patch_tokens = [t.reshape(-1, 768) for t in patch_tokens]
        all_features.append(patch_tokens)
    support_features = [
        torch.cat([all_features[j][i] for j in range(len(all_features))], dim=0)
        for i in range(len(all_features[0]))
    ]
    return support_features


def get_predictions(
    model: nn.Module,
    class_text_embeddings: torch.Tensor,
    test_loader: DataLoader,
    device: str,
    img_size: int,
    dataset: str = "MVTec",
):
    masks = []
    labels = []
    preds = []
    preds_image = []
    file_names = []
    for input_data in tqdm(test_loader):
        image = input_data["image"].to(device)
        mask = input_data["mask"].cpu().numpy()
        label = input_data["label"].cpu().numpy()
        file_name = input_data["file_name"]
        # set up class-specific containers
        class_name = input_data["class_name"]
        assert len(set(class_name)) == 1, "mixed class not supported"
        masks.append(mask)
        labels.append(label)
        file_names.extend(file_name)
        # get text
        epoch_text_feature = class_text_embeddings
        # forward image
        patch_features, det_feature = model(image)
        # calculate similarity and get prediction
        # cls_preds = []
        pred = det_feature @ epoch_text_feature
        pred = (pred[:, 1] + 1) / 2
        preds_image.append(pred.cpu().numpy())
        patch_preds = []
        for f in patch_features:
            # f: bs,patch_num,768
            patch_pred = calculate_similarity_map(
                f, epoch_text_feature, img_size, test=True, domain=DOMAINS[dataset]
            )
            patch_preds.append(patch_pred)
        patch_preds = torch.cat(patch_preds, dim=1).sum(1).cpu().numpy()
        preds.append(patch_preds)
    masks = np.concatenate(masks, axis=0)
    labels = np.concatenate(labels, axis=0)
    preds = np.concatenate(preds, axis=0)
    preds_image = np.concatenate(preds_image, axis=0)
    return masks, labels, preds, preds_image, file_names


def main():
    parser = argparse.ArgumentParser(description="Training")
    # model
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-L-14-336",
        help="ViT-B-16-plus-240, ViT-L-14-336",
    )
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--relu", action="store_true")
    # testing
    parser.add_argument("--dataset", type=str, default="MVTec")
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    # exp
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--save_path", type=str, default="ckpt/baseline")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--text_norm_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_weight", type=float, default=0.1)
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_until", type=int, default=3)
    parser.add_argument("--image_adapt_until", type=int, default=6)

    args = parser.parse_args()
    # ========================================================
    setup_seed(args.seed)
    # check save_path and setting logger
    os.makedirs(args.save_path, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.save_path, "test.log"),
        encoding="utf-8",
        level=logging.INFO,
    )
    logger.info("args: %s", vars(args))
    # set device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # ========================================================
    # load model
    # set up model for testing
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

    # ---- load checkpoints ----
    text_ckpt = os.path.join(args.save_path, "text_adapter.pth")
    image_ckpt = os.path.join(args.save_path, "image_adapter.pth")
    if not os.path.isfile(text_ckpt):
        raise FileNotFoundError(f"Missing {text_ckpt}")
    if not os.path.isfile(image_ckpt):
        raise FileNotFoundError(f"Missing {image_ckpt}")
    text_ckpt = torch.load(text_ckpt, map_location=device)
    model.text_adapter.load_state_dict(text_ckpt["text_adapter"])
    image_ckpt = torch.load(image_ckpt, map_location=device)
    model.image_adapter.load_state_dict(image_ckpt["image_adapter"])
    logger.info("Loaded text_adapter.pth and image_adapter.pth")

    # ========================================================
    # load dataset
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    image_datasets = get_dataset(
        args.dataset,
        args.img_size,
        None,
        args.shot,
        "test",
        logger=logger,
    )
    with torch.no_grad():
        text_embeddings = get_adapted_text_embedding(model, args.dataset, device)

    # ========================================================
    df = DataFrame(
        columns=[
            "class name",
            "pixel AUC",
            "pixel AP",
            "image AUC",
            "image AP",
        ]
    )
    for class_name, image_dataset in image_datasets.items():
        image_dataloader = torch.utils.data.DataLoader(
            image_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
        )

        # ========================================================
        # testing
        with torch.no_grad():
            class_text_embeddings = text_embeddings[class_name]
            masks, labels, preds, preds_image, file_names = get_predictions(
                model=model,
                class_text_embeddings=class_text_embeddings,
                test_loader=image_dataloader,
                device=device,
                img_size=args.img_size,
                dataset=args.dataset,
            )
        # ========================================================
        if args.visualize:
            visualize(
                masks,
                preds,
                file_names,
                args.save_path,
                args.dataset,
                class_name=class_name,
            )
        class_result_dict = metrics_eval(
            masks,
            labels,
            preds,
            preds_image,
            class_name,
            domain=DOMAINS[args.dataset],
        )
        df.loc[len(df)] = Series(class_result_dict)

    metric_cols = ["pixel AUC", "pixel AP", "image AUC", "image AP"]
    df[metric_cols] = df[metric_cols].astype(float)

    avg = df[metric_cols].mean().to_dict()
    avg["class name"] = "Average"
    df.loc[len(df)] = Series(avg)

    logger.info("\n%s", df.to_string(index=False))
    logging.shutdown()


if __name__ == "__main__":
    main()
