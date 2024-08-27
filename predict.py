import re
import os
from os import PathLike
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import yaml
import fire
import torch
import glob

import sys
stdout = sys.stdout

os.environ['TORCH_HOME'] = './models/'

from super_gradients.training import models

sys.stdout = stdout
print("-----------------super_gradients: THE STDOUT PROBLEM STILL EXIST-----------------\n")
print('Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
print("Cuda is available:", torch.cuda.is_available())
print("Cuda version:", torch.version.cuda)
print("Device number:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0))
print("---------------------------------------------------------------------------------\n")
CHECKPOINT_DIR = './checkpoints'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def save_annotations_darknet(image_path, labels, bboxes, confidences, threshold, save_dir):
    image = cv2.imread(str(image_path))
    new_txt_path = f"{save_dir}/{Path(image_path).stem}.txt"
    with open(new_txt_path, "w") as f:
        for label, bbox, conf in zip(labels, bboxes, confidences):
            if conf > threshold:
                x, y, w, h = yolo_to_darknet(image, bbox)
                f.write(str(label)+ " " + str(conf) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h))
                f.write("\n")


def save_annotated_images(image_path, labels, classes, bboxes, confidences, threshold, save_dir):
    image = cv2.imread(str(image_path))
    # h, w = image.shape[:2]
    
    for label, bbox, conf in zip(labels, bboxes, confidences):
        if conf > threshold:
            text = f"{classes[label]} {conf:.2f}"
            COLOR = list(np.random.random(size=3) * 256)
            WHITE_COLOR = (255,255,255)
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), COLOR, 2)
            (w_t, h_t), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1]) - h_t - 6), (int(bbox[0]) + w_t, int(bbox[1])), COLOR, -1)
            # cv2.polylines(image, [mask], True,(0, 255, 255), 3) # draw segmentation
            cv2.putText(image, text, (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE_COLOR, 3)

    cv2.imwrite(f"{save_dir}/{Path(image_path).stem}.jpg", image)


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        path = Path(f"{path}{sep}{n}{suffix}")
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


def yolo_to_darknet(img, rect):
    height, width, _ = img.shape
    x1, y1, x2, y2 = rect
    
    x = ((x2 + x1)/2)/width
    y = ((y2 + y1)/2)/height
    w = (x2 - x1)/width
    h = (y2 - y1)/height
    
    return float(x), float(y), float(w), float(h)


def yolo_to_pascal_voc(bbox, w, h):
    # bbox:[center_x, center_y, width, heigth]
    w_half_len = (float(bbox[2]) * w) / 2
    h_half_len = (float(bbox[3]) * h) / 2
    xmin = int((float(bbox[0]) * w) - w_half_len)
    ymin = int((float(bbox[1]) * h) - h_half_len)
    xmax = int((float(bbox[0]) * w) + w_half_len)
    ymax = int((float(bbox[1]) * h) + h_half_len)
    return [xmin, ymin, xmax, ymax]


def make_gt_yolo_txt(image_path: PathLike, txt_path: PathLike, obj_names, save_path):
    new_txt = Path(save_path) / (Path(txt_path).name)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    height, width = image.shape[0], image.shape[1]
    with open(txt_path, "r") as f:
        lines = f.readlines()

    with open(new_txt, "w") as f:
        for line in lines:
            dn_coords = list(line.split())
            id = int(dn_coords[0])
            label = obj_names[id]
            res = yolo_to_pascal_voc(dn_coords[1:], width, height)
            string_list = list(map(str, res))
            string_line = label + ' ' + ' '.join(string_list)
            f.write(string_line)
            f.write('\n')


def make_det_yolo_txt(image_path: PathLike, txt_path: PathLike, obj_names, save_path):
    new_txt = Path(save_path) / (Path(txt_path).name)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    height, width = image.shape[0], image.shape[1]
    with open(txt_path, "r") as f:
        lines = f.readlines()

    with open(new_txt, "w") as f:
        for line in lines:
            dn_coords = list(line.split())
            id = int(dn_coords[0])
            label = obj_names[id]
            res = yolo_to_pascal_voc(dn_coords[2:], width, height)
            string_list = list(map(str, res))
            string_line = label + ' ' + str(dn_coords[1]) + ' ' + ' '.join(string_list)
            f.write(string_line)
            f.write('\n')


def make(images_path, labels_path, classes, save_path, pref):
    print(f"Save {pref} annotatons to {save_path}")
    imgs_list = []
    txts_list = []
    
    try:
        for txt in labels_path.glob('*.txt'):
            txts_list.append(str(txt))
    except ValueError:
            print("Dataset is Empty!")
    
    types = ["*.jp*g", "*.png"]
    for t in types:
        try:
            for img in images_path.glob(t):
                imgs_list.append(str(img))
        except ValueError:
            print("Dataset is Empty!")
    
    imgs_list.sort()
    txts_list.sort()

    for img, txt in tqdm(zip(imgs_list, txts_list), total=len(imgs_list)):
        if pref == "det":
            make_det_yolo_txt(img, txt, classes, save_path)
        elif pref == "gt":
            make_gt_yolo_txt(img, txt, classes, save_path)


def main(
    dataset_params_dir: PathLike,
    predict_params_dir: PathLike
    ):
    """
        Args:
            dataset_params_dir (Path)
            predict_params_dir (Path)
    """
    classes = yaml.safe_load(open(dataset_params_dir))["classes"]
    num_classes = len(classes)

    global_dataset = yaml.safe_load(open(dataset_params_dir))["global_dataset_params"]

    data_dir = global_dataset["data_dir"]
    test_imgs_dir = Path(data_dir) / global_dataset["test"]["images"]
    test_labels_dir = Path(data_dir) / global_dataset["test"]["labels"]

    predict_config = yaml.safe_load(open(predict_params_dir))["predict_params"]
    checkpoint = predict_config["weights_dir"]

    best_model = models.get(predict_config["model"],
                            num_classes=num_classes,
                            checkpoint_path=checkpoint)

    save_dir = Path(predict_config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_exp_dir = increment_path(save_dir / predict_config["save_dir_name"], 
                              exist_ok=predict_config["exist_ok"]
                            )
    save_exp_dir.mkdir(parents=True, exist_ok=True)
    test_imgs_list = [p for p in Path(test_imgs_dir).iterdir() if p.is_file()]

    for img in tqdm(test_imgs_list):
        predictions = best_model.to(DEVICE).predict(str(img), conf=predict_config["model_conf_thresh"])
        labels = predictions.prediction.labels.tolist()
        bboxes = predictions.prediction.bboxes_xyxy.tolist()
        confidences = predictions.prediction.confidence.tolist()
        classes = predictions.class_names
        save_annotated_images(img, labels, classes, bboxes, confidences, predict_config["conf_thresh"], save_exp_dir)
        save_annotations_darknet(img, labels, bboxes, confidences, predict_config["conf_thresh"], save_exp_dir)
        # predictions.show()
        # predictions.save(output_folder=save_exp_dir)

    save_gt_path = Path(save_exp_dir) / "pascal_voc_gt"
    save_det_path = Path(save_exp_dir) / "pascal_voc_det"
    save_odm_result_path = Path(save_exp_dir) / "ODM_results"
    
    save_gt_path.mkdir(parents=True, exist_ok=True)
    save_det_path.mkdir(parents=True, exist_ok=True)
    save_odm_result_path.mkdir(parents=True, exist_ok=True)
    
    make(test_imgs_dir, test_labels_dir, classes, save_gt_path, "gt")
    make(test_imgs_dir, save_exp_dir, classes, save_det_path, "det")


if __name__ == "__main__":
    fire.Fire(main)
