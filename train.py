import os
from os import PathLike
import yaml
import fire
import torch

import sys
stdout = sys.stdout

os.environ['TORCH_HOME'] = './models/'

from super_gradients.training import models, Trainer
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val

sys.stdout = stdout
print("-----------------super_gradients: THE STDOUT PROBLEM STILL EXIST-----------------\n")
print('Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
print("Cuda is available:", torch.cuda.is_available())
print("Cuda version:", torch.version.cuda)
print("Device number:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0))
print("---------------------------------------------------------------------------------\n")
CHECKPOINT_DIR = './checkpoints'


def main(
    dataset_params_dir: PathLike,
    train_params_dir: PathLike
):
    """
        Args:
            dataset_params_dir (Path)
            train_params_dir (Path)
    """
    classes = yaml.safe_load(open(dataset_params_dir))["classes"]
    global_dataset = yaml.safe_load(open(dataset_params_dir))["global_dataset_params"]
    train_config = yaml.safe_load(open(train_params_dir))["train_params"]
    num_classes = len(classes)

    trainer = Trainer(experiment_name=train_config["experiment_name"], 
                      ckpt_root_dir=CHECKPOINT_DIR)
    
    model_s = models.get(train_config["model"],
                        num_classes=num_classes,
                        pretrained_weights=train_config["pretrained_weights"]
                        )

    train_params = {
        "silent_mode": train_config["silent_mode"],
        "average_best_models": train_config["average_best_models"],
        "warmup_mode": train_config["warmup_mode"],
        "warmup_initial_lr": train_config["warmup_initial_lr"],
        "lr_warmup_epochs": train_config["lr_warmup_epochs"],
        "initial_lr": train_config["initial_lr"],
        "lr_mode": train_config["lr_mode"],
        "cosine_final_lr_ratio": train_config["cosine_final_lr_ratio"],
        "optimizer": train_config["optimizer"],
        "optimizer_params": train_config["optimizer_params"],
        "zero_weight_decay_on_bias_and_bn": train_config["zero_weight_decay_on_bias_and_bn"],
        "ema": train_config["ema"],
        "ema_params": train_config["ema_params"],
        "max_epochs": train_config["max_epochs"],
        "mixed_precision": train_config["mixed_precision"],
        "loss": PPYoloELoss(
            use_static_assigner=train_config["loss"]["use_static_assigner"],
            num_classes=num_classes,
            reg_max=train_config["loss"]["reg_max"]
            ),
        "metric_to_watch": train_config["metric_to_watch"],
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=train_config["valid_metrics_list"]["score_thres"],
                top_k_predictions=train_config["valid_metrics_list"]["top_k_predictions"],
                num_cls=num_classes,
                normalize_targets=train_config["valid_metrics_list"]["normalize_targets"],
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=train_config["post_prediction_callback"]["score_threshold"],
                    nms_top_k=train_config["post_prediction_callback"]["nms_top_k"],
                    max_predictions=train_config["post_prediction_callback"]["max_predictions"],
                    nms_threshold=train_config["post_prediction_callback"]["nms_threshold"]
                )
            )
        ]
    }

    data_dir = global_dataset["data_dir"]
    train_imgs_dir = global_dataset["train"]["images"]
    train_labels_dir = global_dataset["train"]["labels"]

    val_imgs_dir = global_dataset["val"]["images"]
    val_labels_dir = global_dataset["val"]["labels"]

    train_data = coco_detection_yolo_format_train(
        dataset_params={
            "data_dir": data_dir,
            "images_dir": train_imgs_dir,
            "labels_dir": train_labels_dir,
            "classes": classes
        },
        dataloader_params={
            "batch_size": train_config["batch_size"],
            "num_workers": train_config["num_workers"]
        }
    )

    val_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': data_dir,
            'images_dir': val_imgs_dir,
            'labels_dir': val_labels_dir,
            'classes': classes
        },
        dataloader_params={
            "batch_size": train_config["batch_size"],
            "num_workers": train_config["num_workers"]
        }
    )

    train_data.dataset.transforms  # transforms are common image transformations

    trainer.train(model=model_s, 
                  training_params=train_params,
                  train_loader=train_data,
                  valid_loader=val_data)


if __name__ == "__main__":
    fire.Fire(main)
