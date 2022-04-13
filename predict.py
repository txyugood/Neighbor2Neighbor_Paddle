import os
import warnings
from argparse import ArgumentParser

import cv2
import numpy as np
import paddle
from utils import load_pretrained_model
from PIL import Image

from arch_unet import UNet

parser = ArgumentParser()

parser.add_argument('--image_path', type=str, default=None)
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--save_dir', type=str, default='./output/')
warnings.filterwarnings('ignore')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def main():
    args = parser.parse_args()
    model = UNet(3, 3, 48)
    if args.model_path is not None:
        load_pretrained_model(model, args.model_path)
    model.eval()
    im_name = args.image_path.split('/')[-1].split('.')[0]
    noisy_im = pil_loader(args.image_path)
    noisy_im = np.array(noisy_im, dtype=np.float32) / 255.0

    H = noisy_im.shape[0]
    W = noisy_im.shape[1]
    val_size = (max(H, W) + 31) // 32 * 32
    noisy_im = np.pad(
        noisy_im,
        [[0, val_size - H], [0, val_size - W], [0, 0]],
        'reflect')
    noisy_im = noisy_im.transpose([2, 0, 1])
    noisy_im = paddle.to_tensor(noisy_im)
    noisy_im = paddle.unsqueeze(noisy_im, 0)
    with paddle.no_grad():
        prediction = model(noisy_im)
        prediction = prediction[:, :, :H, :W]
    prediction = paddle.transpose(prediction, [0, 2, 3, 1])
    prediction = paddle.clip(prediction, 0, 1)
    prediction = prediction.numpy()
    prediction = prediction.squeeze()
    pred255 = np.clip(prediction * 255.0 + 0.5, 0,
                      255).astype(np.uint8)
    pred255 = Image.fromarray(pred255,"RGB")
    pred255.save(os.path.join(args.save_dir, im_name + "_denoise.png"))


if __name__ == '__main__':
    main()
