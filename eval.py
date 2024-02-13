from metrics import *

from tqdm import tqdm
import yaml

import subprocess
import argparse
import sys
import os
import os.path as osp
from pathlib import Path
import traceback


ap = argparse.ArgumentParser(
        description="Run this script per dataset.")
ap.add_argument('--config', help="May be a single file, e.g., ./configs/exp/example.yaml, "
        "or a directory in which case all .yaml files within it (recursive) will be evaluated.")
ap.add_argument('--methods', '-m', nargs='+',
        choices=['fid', 'blip-vqa', 'clipscore', 'unidet'])
ap.add_argument('--debug', action='store_true')
ap.add_argument('--run_dir')
ap.add_argument('--out_dir')
ap.add_argument('--seed')
ap.add_argument('--results_yaml_name')
ap.add_argument('--t2i_compbench_path', default="../T2I-CompBench")
ap.add_argument('--t2i_compbench_pyexe')
ap.add_argument('--fid_ref_images_dir', default="../dataset/coco/")

args = ap.parse_args()

t2i_compbench_dir = args.t2i_compbench_path
t2i_compbench_pyexe = args.t2i_compbench_pyexe
if not t2i_compbench_pyexe:
    t2i_compbench_pyexe = osp.join(t2i_compbench_dir, ".venv/bin/python")

methods = args.methods

# Collect configs
configs = []
if osp.isfile(args.config):
    configs.append(args.config)
elif osp.isdir(args.config):
    for root, dirs, files in os.walk(args.config):
        for fn in files:
            if osp.splitext(fn)[1].lower() in ['.yaml', 'yml']:
                configs.append(osp.join(root, fn))
else:
    print("Invalid config: '{}'".format(args.config))

print("Collected {} configs: {}".format(len(configs), configs))
print("Evalating on {} metrics: {}".format(methods, len(methods)))

# Start Evaluation
for imeth, meth in enumerate(methods):
    print("Method [{}/{}]: {}".format(imeth+1, len(methods), meth))
    for ic, config_fp in enumerate(configs):
        print("\tConfig [{}/{}]: '{}'".format(ic+1, len(configs), config_fp))
        with open(config_fp) as f:
            config = yaml.safe_load(f)
        dataset = config['dataset']
        baselines = config['baselines']
        for ibl, baseline in enumerate(baselines):
            print("\t\tBaseline [{}/{}]: '{}'\n".format(ibl+1, len(baselines), baseline))
            with open(baseline) as f:
                bl_config = yaml.safe_load(f)
            meth_init_d = {
                'name': bl_config['name'],
                'model': bl_config['model'],
                'dataset': dataset
            }
            if args.run_dir: meth_init_d['run_dir'] = args.run_dir
            if args.out_dir: meth_init_d['out_dir'] = args.out_dir
            if args.seed: meth_init_d['seed'] = args.seed
            if args.results_yaml_name: meth_init_d['results_yaml_name'] = args.results_yaml_name

            if meth == 'fid':
                fid_init_d = {**meth_init_d}
                if args.fid_ref_images_dir:
                    fid_init_d['ref_images_dir'] = args.fid_ref_images_dir
                evaluator = FidScore(**fid_init_d)
            elif meth in ('blip-vqa', 'clipscore', 'unidet'):
                t2icb_init_d = {**meth_init_d}
                t2icb_init_d['score_method'] = meth
                t2icb_init_d['t2icompbench_proj_dir'] = t2i_compbench_dir
                t2icb_init_d['t2icompbench_python_exe'] = t2i_compbench_pyexe
                evaluator = T2ICompBenchScore(**t2icb_init_d)

            if args.debug:
                evaluator.delete_temp_dir_on_exit = False

            with evaluator:
                scores = evaluator.write_scores()
            print("\t\tScores: {}".format(scores))
            print()

