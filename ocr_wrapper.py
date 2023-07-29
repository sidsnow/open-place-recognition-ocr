import os
import os.path as osp
import numpy as np
import cv2
import albumentations as A
from typing import Optional, Union
from mmocr.apis import MMOCRInferencer
from tqdm import tqdm


class OCRWrapper:
    """
    pass a folder to a call function of wrapper to process images with augmentation
    pass a file / np.array / folder / list to OCRWrapper.ocr() to proccess without augmentation
    """
    def __init__(self,
                 det: Optional[str] = 'DBNet',
                 det_weights: Optional[str] = None,
                 rec: Optional[str] = 'SVTR',
                 rec_weights: Optional[str] = None,
                 augment: Optional[list] = [A.Equalize(by_channels=False, p=1),
                                            A.Sharpen(alpha=0.3, lightness=0.8, p=1)]
                 ):
        """
        :param det: name or config file for a mmocr detection model
        :param det_weights: path to detection model weights
        :param rec: name or config file for a mmocr recognition model
        :param rec_weights: path to recognition model weights
        :param augment: inference augmentations. default equalize histogram and sharpen
        """
        self.ocr = MMOCRInferencer(det=det,
                                   det_weights=det_weights,
                                   rec=rec,
                                   rec_weights=rec_weights)
        self.augment = A.Compose(augment)

    def _inputs_to_list(self, inputs):
        if inputs[-4] != '.':
            fls = os.listdir(inputs)
            files = [osp.join(inputs, f) for f in fls]
        elif isinstance(inputs, (tuple, np.ndarray)):
            files = list(inputs)
        else:
            files = [inputs]
        return files

    def __call__(self, inputs, save_vis=False, save_pred=False, out_dir='./vis', pred_score_thr=0.78):
        files = self._inputs_to_list(inputs)

        for file in tqdm(files):
            img = cv2.imread(file)

            inp = self.augment(image=img)['image']
            self.ocr(inp, save_vis=save_vis, save_pred=save_pred, out_dir=out_dir, pred_score_thr=pred_score_thr)