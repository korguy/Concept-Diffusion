from utils.data_util import load_data

import yaml

import os
import os.path as osp
from glob import glob
from pathlib import Path
import shutil
from abc import ABC, abstractmethod
import random
import typing


class Metric(ABC):
    def __init__(self, *, name, model, dataset, run_dir="./run", out_dir="./", seed=None, results_yaml_name="results"):
        self.name = name
        self.model = model
        self.dataset = dataset
        self.root_dir = Path(run_dir, Path(dataset).stem, name)
        self.run_dir = run_dir
        self.out_dir = out_dir
        print("Root dir: '{}'".format(self.root_dir))
        if not self.root_dir.is_dir():
            raise ValueError("Invalid root directory: '{}'".format(self.root_dir))
        if not osp.isdir(out_dir):
            raise ValueError("Invalid out_dir: '{}'".format(out_dir))

        self.__uid = "".join([str(random.randint(0, 9)) for _ in range(10)])
        self.delete_temp_dir_on_exit = True

        # Images
        if seed is not None:
        	self.images = sorted(list(glob(f'{self.root_dir}/*/{seed}.png')))
        else:
        	self.images = sorted(list(glob(f'{self.root_dir}/*/*.png')))
    
        # Load data
        self.prompts = load_data(dataset, model, return_raw_prompts=True)
    
        # Load results.yaml
        self.results_yaml_path = Path(out_dir, results_yaml_name + ".yaml")
        if self.results_yaml_path.is_file():
            with open(self.results_yaml_path) as f:
        	    self.__results_data = yaml.safe_load(f)
        else:
        	self.__results_data = {}
    
    def get_temp_dir(self, clear=False):
        temp_dir = osp.join("tmp", self.__class__.__name__, self.__uid)
        if clear:
            if osp.isdir(temp_dir):
                shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir
    
    @abstractmethod
    def get_scores(self) -> dict:
        pass
    
    def write_scores(self):
        datasetname = Path(self.dataset).stem
        data = self.__results_data
        if datasetname not in data:
        	data[datasetname] = {}

        if self.name not in data[datasetname]:
        	data[datasetname][self.name] = {}

        scores = self.get_scores()
        print("Computed scores: {}".format(", ".join(scores.keys())))
        for score_name, score in scores.items():
            data[datasetname][self.name][score_name] = score

        with open(self.results_yaml_path, 'w') as f:
        	yaml.dump(data, f)

        print("Scores written to: '{}'".format(self.results_yaml_path))

        return scores

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.delete_temp_dir_on_exit and osp.isdir(self.get_temp_dir()):
            shutil.rmtree(self.get_temp_dir())
