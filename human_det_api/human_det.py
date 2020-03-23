from mmdet.apis import init_detector, inference_detector
import mmcv
import os
import numpy as np

class Human_Detector_API():
    """
    score_thr: the thread to selelc the bbo
    return results: numpy array,(n,5), n is the bbox number, [x1,y1,x1,y2,score]
    img_path: str
    """
    def __init__(self,
                score_thr=0.75,
                config_file='configs/faster_rcnn_x101_32x4d_fpn_1x_human_det.py',
                checkpoint_file='checkpoints/faster_rcnn_x101_32x4d_fpn_1x_human_det/epoch_12.pth'):
        self.model = init_detector(config_file,checkpoint_file,device='cuda:0')
        self.score_thr = score_thr
    def detector(self,img_path):
        result = inference_detector(self.model,img_path)
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None

        bboxes = np.vstack(bbox_result)
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        return bboxes


if __name__=="__main__":
    h_detectore = Human_Detector_API()
    bbox_results=h_detectore.detector("/root/3646891459_f7dd21774d_z.jpg")



    