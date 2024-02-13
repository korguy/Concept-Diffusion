from metrics.metrics_common import Metric

from cleanfid import fid

import shutil
import os
import os.path as osp


class FidScore(Metric):
    def __init__(self, *, ref_images_dir="../dataset/coco/val2017", **kw):
        super().__init__(**kw)

        if not osp.isdir(ref_images_dir):
            raise ValueError("Invalid ref_images_dir: '{}'".format(ref_images_dir))
        self.ref_images_dir = ref_images_dir

        self.our_images_dir = self.get_temp_dir(clear=True)
        for idx, img in enumerate(self.images):
            if osp.splitext(img)[1].lower() != ".png":
                raise ValueError("Input images must be PNG: '{}'".format(img))
            shutil.copy(img, osp.join(self.our_images_dir, f"{idx}.png")) 

    def get_scores(self) -> dict:
        score = fid.compute_fid(self.ref_images_dir, self.our_images_dir)
        return {
            "FID": score.item()
        }

