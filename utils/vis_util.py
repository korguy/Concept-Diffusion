import os 
import torch
import math
import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image

from utils.ptp_util import AttentionStore, aggregate_attention

def replace_special_token(text: str):
	text = text.replace("<|startoftext|>", "<S> ")
	text = text.replace("<|endoftext|>", "<E> ")
	return text

def horizontally_merge(images):
	imgs_comb = np.hstack([i for i in images])
	return Image.fromarray(imgs_comb)

def vertically_merge(images):
	imgs_comb = np.vstack([i for i in images])
	return Image.fromarray(imgs_comb)

def merge_result_and_attention(img1, img2, padding=0):
	h = img1.shape[0] if img1.shape[0] > img2.shape[0] else img2.shape[0]
	new_im = Image.new("RGB", (img1.shape[1] + img2.shape[1], h+padding), "White")
	new_im.paste(Image.fromarray(img2), (0,0))
	new_im.paste(Image.fromarray(img1), (img2.shape[1], 0))
	return new_im

def visualize_cross_attention_map(
	controller: AttentionStore,
	image: Image,
	text,
	down=None,
	mid=None,
	up=None,
	out=None
):	
	if down == None and mid == None and up == None:
		image = image.resize((256,256))
	image = np.array(image)
	text = replace_special_token(text)
	text = text.split()
	length = len(text)

	attention_map = controller.get_average_attention()
	def aggreate_imgs(map, level):
		_img = []
		for l in level:
			_map = map[l].detach().cpu()
			res = int(math.sqrt(_map.shape[1]))
			_map = _map.reshape(-1, res, res, _map.shape[-1])
			_map = _map.sum(0) / _map.shape[0]
			__img = []
			for i in range(length):
				__map = _map[:,:,i]
				__map = 255 * __map / __map.max()
				__map = __map.reshape(1,1,res,res)
				__map = __map.cuda()
				__map = F.interpolate(__map, size=256, mode='bilinear')
				__map = __map.cpu()
				# __map = (__map - __map.min()) / (__map.max() -__map.min())
				__map = __map.reshape(256, 256)
				__map = np.array(__map).astype(np.uint8)
				# __map = np.uint8((255 * __map))
				__map = cv2.applyColorMap(np.array(__map), cv2.COLORMAP_MAGMA)
				__map = cv2.cvtColor(np.array(__map), cv2.COLOR_RGB2BGR)
				__img.append(__map)
			_img.append(np.hstack([i for i in __img]))
		return np.vstack([i for i in _img])

	def add_text(img, y_offset):
		x_offset = image.shape[0]
		font = cv2.FONT_HERSHEY_SIMPLEX
		for i, t in enumerate(text):
			textsize = cv2.getTextSize(t, font, 1, 2)[0]
			text_x = x_offset + (i * 256) + ((256-textsize[0]) // 2)
			text_y = y_offset + 100 - textsize[1] // 2
			cv2.putText(img, t, (text_x, text_y), font, 1, (0,0,0), 2)
		return img
	
	img = []
	if down is not None:
		img.append(aggreate_imgs(attention_map['down_cross'][down['head']::2], down['level']))

	if mid is not None:
		img.append(aggreate_imgs(attention_map['mid_cross'][mid['head']::], mid['level']))

	if up is not None:
		img.append(aggreate_imgs(attention_map['up_cross'][up['head']::], up['level']))

	# ADD result and map image
	if len(img) != 0:
		img = np.vstack([i for i in img])

	attention_maps = aggregate_attention(controller, 
		16, 
		("up", "down", "mid"), 
		True, 
		0)

	avg = []
	for i in range(length):
		_map = attention_maps[:, :, i].cpu()
		_map = 255 * _map / _map.max()
		_map = _map.reshape(1,1,16,16)
		_map = _map.cuda()
		_map = F.interpolate(_map, size=256, mode='bilinear')
		_map = _map.cpu()
		_map = _map.reshape(256, 256)
		_map = cv2.applyColorMap(np.array(_map).astype(np.uint8), cv2.COLORMAP_MAGMA)
		_map = cv2.cvtColor(np.array(_map), cv2.COLOR_RGB2BGR)
		avg.append(_map)
	avg = np.hstack([i for i in avg])
	if len(img) != 0:
		img = np.vstack([i for i in [img, avg]])
	else:
		img = avg

	new_img = merge_result_and_attention(img, image, 150)

	# Add Text underneath
	new_img = add_text(np.array(new_img), img.shape[0])
	new_img = Image.fromarray(new_img)
	new_img.save(out)