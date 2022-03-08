import argparse
import os
from glob import glob

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn

from data.dataset import CIFAR10Dataset
from utils.get_path import load_path
from data.data_loader import get_data_loader
from models.model.models import Model
from utils.get_path import get_save_kfold_model_path

from models.model.models import Model
from models.runners.training_runner import TrainingRunner
from models.runners.inference_runner import InferenceRunner
from utils.fix_seed import seed_torch
from utils.translation import str2bool

SAVE_MODEL_NAME = "model.pt"
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


def kfold_main_loop(
    args,
    train_img_paths,
    train_labels,
    valid_img_paths,
    valid_labels,
    test_img_paths,
    fold_num,
):

    train_dataset = CIFAR10Dataset(
        img_paths=train_img_paths,
        labels=train_labels,
        training=True,
        img_size=args.img_size,
        use_augmentation=True,
    )
    valid_dataset = CIFAR10Dataset(
        img_paths=valid_img_paths,
        labels=valid_labels,
        training=True,
        img_size=args.img_size,
        use_augmentation=False,
    )
    test_dataset = CIFAR10Dataset(
        img_paths=test_img_paths,
        labels=None,
        training=False,
        img_size=args.img_size,
        use_augmentation=False,
    )

    # ===========================================================================
    train_data_loader, valid_data_loader, test_data_loader = get_data_loader(
        train_dataset, valid_dataset, test_dataset, args.batch_size, args.num_worker
    )

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
    save_model_path, save_folder_path = get_save_kfold_model_path(
        args.save_path, SAVE_MODEL_NAME, fold_num
    )

    # ===========================================================================
    train_runner = TrainingRunner(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_func=loss_func,
        device=device,
        max_grad_norm=args.max_grad_norm,
    )

    # ===========================================================================
    prev_valid_acc: float = 1e-4
    t_loss, t_acc = [], []
    m_loss, m_acc = [], []
    v_loss, v_acc = [], []

    for epoch in range(args.epochs):
        print(f"Epoch : {epoch + 1}")

        if args.mixup:
            train_loss, train_acc, mixup_loss, mixup_acc = train_runner.run(
                train_data_loader, epoch + 1, mixup=args.mixup
            )
            m_loss.append(mixup_loss)
            m_acc.append(mixup_acc)
            print(f"Mixup loss : {mixup_loss}, Mixup acc : {mixup_acc}")
        else:
            train_loss, train_acc = train_runner.run(
                train_data_loader, epoch + 1, mixup=args.mixup
            )

        t_loss.append(train_loss)
        t_acc.append(train_acc)
        print(f"Train loss : {train_loss}, Train acc : {train_acc}")

        valid_loss, valid_acc = train_runner.run(
            valid_data_loader, epoch + 1, training=False, mixup=False
        )
        v_loss.append(valid_loss)
        v_acc.append(valid_acc)
        print(f"Valid loss : {valid_loss}, Valid acc : {valid_acc}")

        train_runner.save_graph(save_folder_path, t_loss, t_acc, v_loss, v_acc)
        if prev_valid_acc < valid_acc:
            prev_valid_acc = valid_acc
            train_runner.save_model(save_path=save_model_path)
            train_runner.save_result(
                epoch,
                save_folder_path,
                train_loss,
                valid_loss,
                train_acc,
                valid_acc,
                args,
            )

    inference_runner = InferenceRunner(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_func=loss_func,
        device=device,
        max_grad_norm=args.max_grad_norm,
    )
    inference_runner.load_model(save_model_path)
    prediction = inference_runner.run(test_data_loader)
    print("len(prediction) :", len(prediction))
    print("Done.")

    decoded_prediction = [LABEL_DECODE_DICT[int(pred)] for pred in prediction]

    submission = pd.read_csv(SUBMISSION_CSV_PATH)
    submission["target"] = decoded_prediction

    print("Save submission")
    save_csv_path = os.path.join(save_folder_path, f"fold{fold_num+1}_submission.csv")
    submission.to_csv(save_csv_path, index=False)
    print("Done.")


if __name__ == "__main__":
    print(len(LABEL_DICT))
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
    args.add_argument("--epochs", type=int, default=300)
    args.add_argument("--max_grad_norm", type=float, default=1.0)
    args.add_argument("--img_size", type=int, default=224)
    args.add_argument("--fc", type=int, default=2048)
    args.add_argument("--num_worker", type=int, default=8)
    args.add_argument("--train_data_path", type=str, default="dataset/train/")
    args.add_argument("--test_data_path", type=str, default="dataset/test/")
    args.add_argument("--save_path", type=str, default="./models/saved_model/")
    args.add_argument("--model_name", type=str, default="timm_classification")
    args.add_argument("--label_smoothing", type=float, default=0.1)
    args.add_argument("--device", type=int, default=0)
    args.add_argument("--mixup", type=str2bool, default="True")

    args = args.parse_args()

    print(args)

    seed_torch(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===========================================================================
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    num_folder = len(glob(args.save_path + "*"))
    args.save_path = os.path.join(args.save_path, str(num_folder + 1))

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # ===========================================================================
    img_paths, labels = load_path(args.train_data_path)

    test_img_paths = load_path(args.test_data_path, train=False)

    print(f"img_paths : {len(img_paths)}")
    print(f"labels: {len(labels)}")

    print(img_paths[0])
    print(f"labels : {labels}")

    fold_list = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train, valid in skf.split(img_paths, labels):
        fold_list.append([train, valid])
        print("train", len(train), train)
        print("valid", len(valid), valid)
        print()

    for fold_num, fold in enumerate(fold_list):
        print(f"Fold num : {str(fold_num + 1)}, fold : {fold}")
        train_img_paths = [img_paths[i] for i in fold[0]]
        train_labels = [labels[i] for i in fold[0]]

        valid_img_paths = [img_paths[i] for i in fold[1]]
        valid_labels = [labels[i] for i in fold[1]]

        kfold_main_loop(
            args,
            train_img_paths,
            train_labels,
            valid_img_paths,
            valid_labels,
            test_img_paths,
            fold_num,
        )
