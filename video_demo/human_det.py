from mmdet.apis import init_detector, inference_detector
import mmcv
import os
import numpy as np



# TODO: merge this method with the one in BaseDetector
def show_result(img,
                result,
                class_names,
                score_thr=0.8,
                wait_time=0,
                show=True,
                out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        wait_time (int): Value of waitKey param.
        show (bool, optional): Whether to show the image with opencv or not.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    ids = labels==0
    labels = labels[ids]
    bboxes = bboxes[ids]
    
    
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        np.random.seed(42)
        color_masks = [
            np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            for _ in range(max(labels) + 1)
        ]
        for i in inds:
            i = int(i)
            color_mask = color_masks[labels[i]]
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # draw bounding boxes
    mmcv.imshow_det_bboxes(
        img,
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show=show,
        wait_time=wait_time,
        out_file=out_file)
    if not (show or out_file):
        return img

class Human_detector():
    def __init__(self,config_file='configs/faster_rcnn_x101_32x4d_fpn_1x.py',checkpoint_file='checkpoints/faster_rcnn_x101_32x4d_fpn_1x_20181218-ad81c133.pth'):
        self.model = init_detector(config_file,checkpoint_file,device='cuda:0')


    def image_det(self,imgs,out_dir):
        files = os.listdir(imgs)
        for f in files:
            if f.endswith(".png") or f.endswith(".jpg"):
                name = os.path.join(imgs,f)
                result = inference_detector(self.model,name)
                out_name = os.path.join(out_dir,f)
                show_result(name,result,self.model.CLASSES,show=False,out_file=out_name)

if __name__ =="__main__":
    detector = Human_detector()
    detector.image_det("/root/studio_frames","/root/tmp_out/")

