from glob import glob
from sklearn.preprocessing import LabelEncoder
import os

from glob import glob
from sklearn.preprocessing import LabelEncoder


def load_path(dataset_path, train=True):
    img_paths = []
    labels = []

    if train:
        for path in glob(dataset_path + "*"):
            img_path = sorted(glob(path + "/*.jpg"))
            img_paths.extend(img_path)

            labels = [x.split("/")[2] for x in img_paths]
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        return img_paths, labels
    else:
        img_paths = sorted(glob(dataset_path + "/*.jpg"))
        return img_paths


def get_save_kfold_model_path(save_path: str, save_model_name: str, fold_num: int):
    # fold 저장할 폴더
    save_folder_path = os.path.join(save_path, str(fold_num + 1))

    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)

    save_model_path = os.path.join(save_folder_path, save_model_name)
    print(f"Model Save Path : {save_folder_path}")

    return save_model_path, save_folder_path
