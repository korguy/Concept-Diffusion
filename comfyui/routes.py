from .common import CONFIGS_DIR, extract_nouns
from server import PromptServer

from aiohttp import web
import yaml

import os
import os.path as osp
from pathlib import Path
import traceback
import json


routes = PromptServer.instance.routes


@routes.post("/concept-guidance/load-config")
async def load_config_endpoint(req):
    post = await req.post()
    if not post.get('model_name', None):
        return web.Response(400, text="Request lacks the 'model_name' field.")
    config_fn = post.get("config_filename")
    if not config_fn:
        print("config_filename is empty")
        return web.Response(status=400)
    config_path = Path(CONFIGS_DIR, config_fn)
    if not config_path.is_file():
        print("Config file not found: '{}'".format(config_path))
        return web.Response(status=400)

    with open(config_path, encoding='utf-8') as f:
        try:
            conf = yaml.safe_load(f)
        except yaml.YAMLError:
            print("Failed loading conf: '{}'".format(config_path))
            traceback.print_exc()
            return web.Response(status=400)
    print("Loaded config (for '{}'): '{}'".format(post['model_name'], config_path))

    if not conf.get('model', None):
        print("Invalid config; config lacks 'model' field.")
        return web.Response(status=400)
    if conf['model'] != post['model_name']:
        return web.Response(status=400,
                text="This configuration is for '{}' not '{}'.".format(conf['model'], post['model_name']))

    params_flat = {**conf.get('params', {})}
    for k, v in params_flat.items():
        if type(v) not in (int, float, bool, str):
            params_flat[k] = json.dumps(v)
    if 'params' in conf:
        conf['params'] = params_flat

    ret = {
            'config': conf
    }
    return web.json_response(ret)


@routes.post("/concept-guidance/extract-nouns")
async def extract_nouns_endpoint(req):
    post = await req.post()
    nouns = None
    if post.get('prompt', None):
        prompt = post['prompt']
        nouns = extract_nouns(prompt)
    return web.json_response({
            "nouns": nouns or [],
        })

