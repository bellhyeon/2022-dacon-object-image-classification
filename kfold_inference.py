"""Kfold Inference"""
import argparse
import os
from glob import glob
from time import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from data.dataset import CIFAR10Dataset
from models.model.models import Model
from models.runners.inference_runner import InferenceRunner
from utils.fix_seed import seed_torch

from utils.get_path import load_path

LABEL_DICT = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9,
}
LABEL_DECODE_DICT = {v: k for k, v in LABEL_DICT.items()}
SUBMISSION_CSV_PATH = "dataset/sample_submission.csv"
TEST_IMAGE_PATH = "dataset/test/"


def get_inference_runner(args, save_model_path):
    # ===========================================================================
    model = Model.get_model(args.model_name, args.__dict__).to(args.device)

    # ===========================================================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    loss_func = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.T_max, eta_min=args.eta_min
    )

    # ===========================================================================
    inference_runner = InferenceRunner(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_func=loss_func,
        device=args.device,
        max_grad_norm=args.max_grad_norm,
    )
    inference_runner.load_model(save_model_path)
    return inference_runner


def kfold_main_loop(
    args,
    test_img_paths,
    save_model_path,
):

    test_dataset = CIFAR10Dataset(
        img_paths=test_img_paths,
        training=False,
        labels=None,
        img_size=args.img_size,
        use_augmentation=False,
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # ===========================================================================
    # pylint: disable=invalid-name
    model = Model.get_model(args.model_name, args.__dict__).to(device)

    # ===========================================================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    loss_func = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.T_max, eta_min=args.eta_min
    )

    # ===========================================================================
    inference_runner = InferenceRunner(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_func=loss_func,
        device=device,
        max_grad_norm=args.max_grad_norm,
    )
    inference_runner.load_model(save_model_path)
    prediction = inference_runner.infer(test_data_loader)
    print("len(prediction) :", prediction.shape)
    print("Done.")

    return prediction


def save_submission(prediction, save_folder_path):
    decoded_prediction = [LABEL_DECODE_DICT[int(pred)] for pred in prediction]

    submission = pd.read_csv(SUBMISSION_CSV_PATH)
    submission["target"] = decoded_prediction

    print("Save submission")
    save_csv_path = os.path.join(save_folder_path, "kfold_result.csv")
    submission.to_csv(save_csv_path, index=False)
    print("Done.")


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--backbone", type=str, default="resnet34d")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--batch_size", type=int, default=128)
    args.add_argument("--num_classes", type=int, default=len(LABEL_DICT))
    args.add_argument("--T_max", type=int, default=10)
    args.add_argument("--eta_min", type=float, default=1e-6)
    args.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    args.add_argument("--betas", type=tuple, default=(0.9, 0.999))
    args.add_argument("--eps", type=float, default=1e-8)
    args.add_argument("--weight_decay", type=float, default=1e-3)
    args.add_argument("--epochs", type=int, default=50)
    args.add_argument("--max_grad_norm", type=float, default=1.0)
    args.add_argument("--img_size", type=int, default=224)
    args.add_argument("--fc", type=int, default=2048)
    args.add_argument("--num_worker", type=int, default=0)
    args.add_argument("--model_name", type=str, default="timm_classification")
    args.add_argument("--label_smoothing", type=float, default=0.1)
    args.add_argument("--device", type=int, default=0)
    args.add_argument(
        "--save_model_path",
        type=str,
        default="models/saved_model/1/*/model.pt",
    )
    args.add_argument(
        "--save_folder_path",
        type=str,
        default="models/saved_model/1/",
    )

    args = args.parse_args()

    infer_start_time = time()
    seed_torch(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # ===========================================================================
    test_img_paths = load_path(TEST_IMAGE_PATH, train=False)

    print(f"test_img_paths : {len(test_img_paths)}")

    infer_results = []

    save_model_paths = sorted(glob(args.save_model_path))
    print("saved model paths", save_model_paths)

    for fold_num, save_model_path in enumerate(save_model_paths):
        print("=" * 100)
        print(f"Model trained fold : {fold_num + 1}")
        print(f"Saved Model path : {save_model_path}")

        infer_result = kfold_main_loop(
            args,
            test_img_paths,
            save_model_path,
        )

        infer_results.append(infer_result)

    print("Soft Voting")
    prediction = (
        infer_results[0]
        + infer_results[1]
        + infer_results[2]
        + infer_results[3]
        + infer_results[4]
    )
    prediction = prediction / 5
    prediction = [np.argmax(i) for i in prediction]

    save_submission(prediction, args.save_folder_path)
    # ===========================================================================

    print(f"Inference Time : {time() - infer_start_time} sec")
