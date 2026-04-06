from medmnist import INFO
import medmnist
import os


def download_data(data_flag, root):
    os.makedirs(root, exist_ok=True)

    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])

    for split in ["train", "val", "test"]:
        print(f"Downloading {split}...")
        DataClass(split=split, root=root, download=True)


if __name__ == "__main__":
    from src.common.config import load_config

    config = load_config("config.yaml")
    download_data(config["data"]["data_flag"], config["data"]["raw_dir"])
    print("Data download complete.\n")