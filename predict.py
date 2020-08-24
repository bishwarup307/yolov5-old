"""
__author__: bishwarup
created: Sunday, 26th July 2020 12:18:05 am
"""

from typing import Union, List, Optional, Tuple
import os
import json
import cv2
from pathlib import Path
import numpy as np
import argparse
import time
from colorama import Fore, Style
import glob
from tqdm import tqdm
from joblib import Parallel, delayed
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from utils.utils import non_max_suppression, check_img_size
from models.experimental import attempt_load, Ensemble


def restricted_float(x):
    """utility for command line options
    """
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x


class YoloDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[str, os.PathLike],
        file_ext: Optional[str] = "jpg",
        resize: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        self._paths = glob.glob(os.path.join(data_dir, "*." + file_ext))
        self.resize = None
        if resize is not None:
            if not hasattr(resize, "__len__"):
                self.resize = (resize, resize)
            else:
                assert len(resize) == 2
                self.resize = resize

    @property
    def filenames(self):
        filenames = [os.path.basename(f) for f in self._paths]
        return filenames

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        imgf = random.choice(self._paths) if idx > len(self._paths) else self._paths[idx]
        img = cv2.imread(imgf)
        if self.resize is not None:
            img = cv2.resize(img, self.resize, interpolation=cv2.INTER_CUBIC)
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        return img


def postprocess(pred):
    if pred is None:
        return []
    bboxes = pred[:, :4].astype(int).tolist()
    confs = pred[:, 4].tolist()
    pred_classes = pred[:, 5].astype(int).tolist()
    preds = [
        {"class_index": int(cl), "bbox": bbox, "confidence": float(conf)}
        for cl, bbox, conf in zip(pred_classes, bboxes, confs)
    ]
    return preds


def to_cpu(x):
    try:
        x = x.cpu().numpy()
    except AttributeError:
        x = None
    return x


def predict(
    data_dir: Union[str, os.PathLike],
    weights: Union[str, os.PathLike],
    batch_size: Optional[int] = 8,
    num_workers: Optional[int] = 1,
    resize: Optional[Union[int, Tuple[int, int]]] = None,
    file_ext: Optional[str] = "jpg",
    confidence: Optional[float] = 0.5,
    nms_threshold: Optional[float] = 0.5,
    output_path: Union[str, os.PathLike] = "../",
):
    dataset = YoloDataset(data_dir, file_ext=file_ext, resize=resize)
    loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device {device}")

    print("loading model...")
    model = attempt_load(weights, map_location=device)
    model.eval()
    # print(type(model))

    predictions = []
    for batch in tqdm(loader):
        batch = batch.to(device)
        # print(batch.size())
        with torch.no_grad():
            pred = model(batch, augment=False)[0]
        pred = non_max_suppression(pred, confidence, nms_threshold, classes=None, agnostic=False)
        predictions.extend([to_cpu(p) for p in pred])

    predictions = Parallel(n_jobs=os.cpu_count(), backend="multiprocessing")(
        delayed(postprocess)(p) for p in tqdm(predictions)
    )

    if output_path.endswith(".json"):
        if os.path.exists(os.path.dirname(output_path)):
            output_file = output_path
        else:
            raise IOError(
                f"{Fore.RED} no such directory {os.path.dirname(output_path)} {Style.RESET_ALL}"
            )
    elif os.path.isdir(output_path):
        output_file = os.path.join(
            output_dir, "yolov5_predictions_" + str(time.time()).split(".")[0] + ".json"
        )

    else:
        raise IOError(
            f"{Fore.RED} no such directory {os.path.dirname(output_path)} {Style.RESET_ALL}"
        )

    filenames = dataset.filenames
    output_dict = dict(zip(filenames, predictions))

    with open(output_file, "w") as f:
        json.dump(output_dict, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="yolov5 prediction")
    parser.add_argument("-i", "--image-dir", help="path to the directory with test images")
    parser.add_argument("-w", "--weights", help="path to the weights file")
    parser.add_argument("-o", "--output-path", help="path to output file/directory")
    parser.add_argument(
        "--batch-size", type=int, default=8, required=False, help="batch size for prediction"
    )
    parser.add_argument(
        "--confidence",
        type=restricted_float,
        required=False,
        default=0.5,
        help="minimum confidence score for the predictions",
    )
    parser.add_argument(
        "--nms-threshold",
        type=restricted_float,
        required=False,
        default=0.5,
        help="IOU threshold for non-max-suppression",
    )
    parser.add_argument(
        "--resize",
        required=False,
        help="resize the images to a specified dimension before predictions",
    )
    parser.add_argument(
        "--file-ext", required=False, default="jpg", help="file extension of the images"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        required=False,
        default=1,
        help="number of workers for multiprocessing",
    )

    opt = parser.parse_args()
    resize = opt.resize
    if resize is not None:
        resize = tuple(map(int, resize.split(",")[:2]))
        if len(resize) == 1:
            resize = resize[0]

    predict(
        opt.image_dir,
        opt.weights,
        opt.batch_size,
        opt.num_workers,
        resize,
        opt.file_ext,
        opt.confidence,
        opt.nms_threshold,
        opt.output_path,
    )
