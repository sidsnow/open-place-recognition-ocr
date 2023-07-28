from mmocr.apis import TextDetInferencer, TextRecInferencer
from mmocr.structures import TextSpottingDataSample
from mmocr.visualization import TextSpottingLocalVisualizer
from mmocr.utils import bbox2poly, crop_img, poly2bbox
import cv2
import numpy as np
import os
import os.path as osp
from tqdm import tqdm
from typing import Optional, Dict, List


class Inferencer:
    def __init__(self, det=None, det_weights=None, rec=None, rec_weights=None):
        super().__init__()
        self.det = TextDetInferencer(model=det, weights=det_weights)
        self.rec = TextRecInferencer(model=rec, weights=rec_weights)
        self.vis = TextSpottingLocalVisualizer()

    def _pack_sample(self, preds: Dict) -> List[TextSpottingDataSample]:

        results = []

        det_data_sample, rec_data_samples = preds['det'], preds['rec']
        texts = []
        for rec_data_sample in rec_data_samples:
            texts.append(rec_data_sample.pred_text.item)
        det_data_sample.pred_instances.texts = texts
        results.append(det_data_sample)
        return results

    def process_folder(self,
                       root_dir: str,
                       out_dir: Optional[str] = None) -> dict:
        result = {'det': [], 'rec': []}

        inputs = os.listdir(root_dir)
        for file in tqdm(inputs):

            if self.det:
                result['det'].append(
                    self.det(sharpened, return_datasamples=True)['predictions']
                )
            if self.rec:
                det_res = result['det'][:-1]
                print(result['det'])
                quad = bbox2poly(poly2bbox(det_res['polygon']))
                to_res = crop_img(sharpened, quad)
                result['rec'].append(
                    self.rec(to_res, return_datasamples=True)['predictions']
                )
            preds = self._pack_sample(list(result.items())[-1])
            self.save_visualization(file, img, preds, out_dir)
        return result

    def save_visualization(self, img_name, img, preds, out_dir, draw_pred=True, pred_score_thr=0.75):
        img_out_dir = osp.join(out_dir, 'vis')

        out_file = osp.splitext(img_name)[0]
        out_file = f'{out_file}.jpg'
        out_file = osp.join(img_out_dir, out_file)

        self.vis.add_datasample(
            img_name,
            img,
            preds,
            draw_pred=draw_pred,
            pred_score_thr=pred_score_thr,
            out_file=out_file,
        )