import os
import argparse

import paddle
import numpy as np

from dataset import validation_bsd300
from arch_unet import UNet
from utils import calculate_psnr, calculate_ssim, load_pretrained_model

parser = argparse.ArgumentParser()
parser.add_argument('--val_dirs', type=str, default='./validation')
parser.add_argument('--model_path', type=str, default='./validation')
parser.add_argument('--n_feature', type=int, default=48)
parser.add_argument('--n_channel', type=int, default=3)
opt, _ = parser.parse_known_args()

BSD300_dir = os.path.join(opt.val_dirs, "BSD300")
valid_dict = {
    "BSD300": validation_bsd300(BSD300_dir)
}

network = UNet(in_nc=opt.n_channel,
               out_nc=opt.n_channel,
               n_feature=opt.n_feature)
load_pretrained_model(network, opt.model_path)
network.eval()

np.random.seed(101)
# valid_repeat_times = {"Kodak": 10, "BSD300": 3, "Set14": 20}
valid_repeat_times = {"BSD300": 3}

for valid_name, valid_images in valid_dict.items():
    psnr_result = []
    ssim_result = []
    repeat_times = valid_repeat_times[valid_name]
    for i in range(repeat_times):
        for idx, im in enumerate(valid_images):
            origin255 = im.copy()
            origin255 = origin255.astype(np.uint8)
            im = np.array(im, dtype=np.float32) / 255.0
            noisy_im = np.array(im + np.random.normal(size=im.shape) * (25 / 255),
                            dtype=np.float32)
            if i == 0 and idx < 5:
                noisy255 = noisy_im.copy()
                noisy255 = np.clip(noisy255 * 255.0 + 0.5, 0,
                                   255).astype(np.uint8)
            # padding to square
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
                prediction = network(noisy_im)
                prediction = prediction[:, :, :H, :W]
            # prediction = prediction.permute(0, 2, 3, 1)
            prediction = paddle.transpose(prediction, [0, 2, 3, 1])
            prediction = paddle.clip(prediction, 0, 1)
            prediction = prediction.numpy()
            prediction = prediction.squeeze()
            pred255 = np.clip(prediction * 255.0 + 0.5, 0,
                              255).astype(np.uint8)
            # calculate psnr
            cur_psnr = calculate_psnr(origin255.astype(np.float32),
                                      pred255.astype(np.float32))
            psnr_result.append(cur_psnr)
            cur_ssim = calculate_ssim(origin255.astype(np.float32),
                                      pred255.astype(np.float32))
            ssim_result.append(cur_ssim)

    psnr_result = np.array(psnr_result)
    avg_psnr = np.mean(psnr_result)
    avg_ssim = np.mean(ssim_result)
    print(f"[EVAL] {valid_name}: psnr:{avg_psnr} ssim:{avg_ssim}")
