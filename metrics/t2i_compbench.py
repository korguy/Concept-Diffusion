from metrics.metrics_common import Metric

from PIL import Image, ExifTags
from tqdm import tqdm

import os
import os.path as osp
import shutil
import errno
from pathlib import Path
import datetime
import json
import csv
import re
import subprocess
import operator


def now_fmt():
    return datetime.datetime.now().strftime("%Y%m%dT%H%M%S")


class T2ICompBenchScore(Metric):
    SCORE_METHODS = ['blip-vqa', 'clipscore', 'unidet', 'minigpt4']

    def __init__(self, *, name, score_method, t2icompbench_proj_dir, t2icompbench_python_exe, **kw):
        super().__init__(name=name, **kw)
        self.score_method = score_method
        self.t2icb_proj_dir = Path(t2icompbench_proj_dir)
        self.t2icb_py_exe = Path(t2icompbench_python_exe)
        if score_method not in self.SCORE_METHODS:
            raise ValueError("Invalid score method: '{}'".format(score_method))
        if not self.t2icb_proj_dir.is_dir() or not self.t2icb_py_exe.is_file():
            raise ValueError("Invalid T2ICompBench config")

        temp_dir = self.get_temp_dir(clear=True)
        self.__img_og_paths = dict()
        self.__t2i_data_dir = Path(temp_dir, "data")
        self.__t2i_data_dir.mkdir()
        self.__samples_dir = Path(self.__t2i_data_dir, "samples")
        self.__samples_dir.mkdir()
        self.__prepare_images()

        self.__results_basedir = Path(self.out_dir, "t2icompbench_results", self.name, score_method, now_fmt())
        self.__results_basedir.mkdir(parents=True, exist_ok=True)
        print("Results will be saved to '{}'.".format(self.__results_basedir))

    def __prepare_images(self):
        for iimg, img in enumerate(tqdm(self.images, desc="Preparing images for T2ICB")):
            prompt_idx = int(re.match(r"^(\d+)_.+$", Path(img).parent.name).group(1))
            caption = self.prompts[prompt_idx]

            target_fn = "{}_{:06}{}".format(caption, iimg, osp.splitext(img)[1])
            target_fp = osp.join(self.__samples_dir, target_fn)
            try:
                os.symlink(osp.abspath(img), target_fp)
            except OSError as e:
                if e.errno == errno.ENAMETOOLONG:
                    img_pil = Image.open(img)
                    exifdata = img_pil.getexif()
                    exifdata[list(ExifTags.TAGS.keys())[list(ExifTags.TAGS.values()).index("UserComment")]] = caption
                    target_fn = "{}CAPTIONTOOLONG_{:06}{}".format(caption.strip()[:150], iimg, osp.splitext(img)[1])
                    target_fp = osp.join(self.__samples_dir, target_fn)
                    img_pil.save(target_fp, exif=exifdata)
                else:
                    raise RuntimeError(e)
            except Exception as e:
                raise RuntimeError(e)
            self.__img_og_paths[target_fp] = img

    def get_scores(self) -> dict:
        og_cwd = osp.abspath(os.getcwd())
        if self.score_method == 'minigpt4':
            t2i_wd = osp.abspath(mgpt4_proj_dir)
        else:
            t2i_wd = osp.abspath(osp.join(self.t2icb_proj_dir,
                {'blip-vqa': "BLIPvqa_eval", 'clipscore': "CLIPScore_eval", 'unidet': "UniDet_eval",
                    '3-in-1': "3_in_1_eval", 'minigpt4': "MiniGPT4-CoT_eval"}[self.score_method]))

        cmd = self.__get_eval_command()
        print("T2ICompBench command to be executed: {}".format(cmd))
        os.chdir(t2i_wd)
        subprocess.call(cmd)
        os.chdir(og_cwd)

        if self.score_method == 'blip-vqa':
            score = self.__get_score_blipvqa()
        elif self.score_method == 'clipscore':
            score = self.__get_score_clipscore()
        elif self.score_method == 'unidet':
            score = self.__get_score_unidet()
        else:
            raise NotImplementedError()
        return {self.score_method: score}

    def __get_eval_command(self):
        pyexe = str(self.t2icb_py_exe.absolute())
        datadir = str(self.__t2i_data_dir.absolute())
        if self.score_method == 'blip-vqa':
            cmd = [pyexe, "BLIP_vqa.py", "--out_dir", datadir, "--np_num", "-1"]
        elif self.score_method == 'clipscore':
            cmd = [pyexe, "CLIP_similarity.py", "--outpath", datadir]
        elif self.score_method == 'unidet':
            cmd = [pyexe, "determine_position_for_eval.py", "--outpath", datadir]
        elif self.score_method == 'minigpt4':
            #cmd = [mgpt4_python_exe, "mGPT_cot_general.py", "--img_file", osp.join(t2i_data_dir, "samples"), "--output_path", mgpt4_out_dir,
            #        "--cfg-path", "./eval_configs/minigpt4_eval.yaml", "--category", "complex"]
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        return cmd

    def __get_score_blipvqa(self):
        test_cfg = {}
        with open(osp.join(self.__t2i_data_dir, "annotation_blip", "vqa_result.json")) as f:
            vqa_result = json.load(f)
        score = sum(map(lambda r: float(r['answer']), vqa_result)) / len(vqa_result)
        shutil.copytree(osp.join(self.__t2i_data_dir, "annotation_blip"), osp.join(self.__results_basedir, "annotation_blip"))

        rows = []
        for result_dir_fn in os.listdir(self.__t2i_data_dir):
            result_dir_fp = osp.join(self.__t2i_data_dir, result_dir_fn)
            if re.match(r"annotation\d_blip", result_dir_fn):
                shutil.copytree(result_dir_fp, osp.join(self.__results_basedir, result_dir_fn))

                cfg_fp = osp.join(result_dir_fp, "color_test.json")
                with open(cfg_fp) as f:
                    cfg_i = json.load(f)
                with open(osp.join(result_dir_fp, "VQA/result/vqa_result.json")) as f:
                    vqa_result_i = json.load(f)

                for r in vqa_result_i:
                    cfg = None
                    for d in cfg_i:
                        if d['question_id'] == r['question_id']:
                            cfg = d
                            break
                    else:
                        print("Could not identify entry with question_id='{}' from '{}'.".format(
                            r['question_id'], cfg_fp), file=sys.stderr)
                        continue
                    score_q = r['answer']
                    rows.append([self.__img_og_paths.get(cfg['image'], cfg['image']), cfg['question'], float(score_q)])

        with open(osp.join(self.__results_basedir, "annotation_blip.csv"), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for r in sorted(rows, key=operator.itemgetter(0)):
                writer.writerow(r)
        return score

    def __get_score_clipscore(self):
        with open(osp.join(self.__t2i_data_dir, "annotation_clip/vqa_result.json")) as f:
            vqa_result = json.load(f)
        score = sum(map(lambda r: float(r['answer']), vqa_result)) / len(vqa_result)
        shutil.copytree(osp.join(self.__t2i_data_dir, "annotation_clip"),
                osp.join(self.__results_basedir, "annotation_clip"))
        return score

    def __get_score_unidet(self):
        with open(osp.join(self.__t2i_data_dir, "labels/annotation_obj_detection/vqa_result.json")) as f:
            vqa_result = json.load(f)
        score_sum = 0
        cnt = 0
        for r in vqa_result:
            score_cur = float(r['answer'])
            if score_cur >= 0:
                score_sum += score_cur
                cnt += 1

        print("Positional information was found in {} out of {} input image captions.".format(cnt, len(vqa_result)))
        if cnt > 0:
            score = score_sum / cnt
        else:
            score = -1
            print("Error: no score calculated")
        return score

