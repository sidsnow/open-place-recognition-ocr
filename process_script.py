import os.path as osp
from ocr_wrapper import OCRWrapper

ocrw = OCRWrapper(det='DBNet',
                  rec=r'add path to config',
                  rec_weights=r'./epoch_20.pth')

root_dir = r'name a !folder! to process'
ocrw(root_dir,
     save_vis=True,
     save_pred=False,
     out_dir=osp.join('./vis1', root_dir))