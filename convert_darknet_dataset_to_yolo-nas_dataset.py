from pathlib import Path
from tqdm import tqdm
import shutil


def copy_images_from_txt(txt_file, dataset_path):
    proj_path = Path(txt_file).parents[2]

    with open(txt_file, "r") as f:
        lines = f.read().splitlines()
        for l in tqdm(lines, desc='Copy'):

            line = proj_path / Path(l)
            # print(line)
            if len(str(line).strip()) == 0:
                print(f"found an end of line {lines.index(line)}")
            else:
                image_path = Path(line)
                # dataset_root = image_path.parents[2]
                folder_name = image_path.parent.name
                image_name = image_path.name
                new_image_name = folder_name + "_" + image_name
                p = Path(dataset_path / new_image_name)
                p = str(Path(__file__).resolve().parent) + str(p)
                shutil.copyfile(line, p)


def copy_labels_from_txt(txt_file, dataset_path):
    proj_path = Path(txt_file).parents[2]

    with open(txt_file, "r") as f:
        lines = f.read().splitlines()
        for l in tqdm(lines, desc='Copy'):

            line = proj_path / Path(l)

            if len(str(line).strip()) == 0:
                print(f"found an end of line {lines.index(line)}")
            else:
                image_path = Path(line)
                txt_path = image_path.with_suffix('.txt')
                # dataset_root = txt_path.parents[2]
                folder_name = txt_path.parent.name
                txt_name = txt_path.name
                new_txt_name = folder_name + "_" + txt_name
                p = Path(dataset_path / new_txt_name)
                p = str(Path(__file__).resolve().parent) + str(p)
                shutil.copyfile(txt_path, p)


def main():
    root = Path(__file__).resolve().parent
    # train_set_txt = "/home/alexsh/darknet_experiments/data_lpr/prepared/train_ptl_458_459_461_462.txt"
    # val_set_txt = "/home/alexsh/darknet_experiments/data_lpr/prepared/valid.txt"
    # test_set_txt = "/home/alexsh/darknet_experiments/data_lpr/prepared/test_ptl_458_459_461_462.txt"
    test_set_txt = "/home/alexsh/darknet_experiments/data_lpr/prepared/test_ptl_458_459.txt"

    # train_images_path = Path("/data/train/images")
    # train_labels_path = Path("/data/train/labels")
    # val_images_path = Path("/data/val/images")
    # val_labels_path = Path("/data/val/labels")
    test_images_path = Path("/data/test_ptl_458_459/images")
    test_labels_path = Path("/data/test_ptl_458_459/labels")

    # Path((str(root) + str(train_images_path))).mkdir(parents=True, exist_ok=True)
    # Path((str(root) + str(train_labels_path))).mkdir(parents=True, exist_ok=True)
    # Path((str(root) + str(val_images_path))).mkdir(parents=True, exist_ok=True)
    # Path((str(root) + str(val_labels_path))).mkdir(parents=True, exist_ok=True)
    Path((str(root) + str(test_images_path))).mkdir(parents=True, exist_ok=True)
    Path((str(root) + str(test_labels_path))).mkdir(parents=True, exist_ok=True)

    # copy_images_from_txt(train_set_txt, train_images_path)
    # copy_labels_from_txt(train_set_txt, train_labels_path)
    # copy_images_from_txt(val_set_txt, val_images_path)
    # copy_labels_from_txt(val_set_txt, val_labels_path)
    copy_images_from_txt(test_set_txt, test_images_path)
    copy_labels_from_txt(test_set_txt, test_labels_path)


if __name__ == "__main__":
    main()
