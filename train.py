from __future__ import division
import os
import time
import glob
import datetime
import argparse

import numpy as np
import paddle
from paddle.io import DataLoader
import cv2
from PIL import Image

from arch_unet import UNet
# from augment_noise import AugmentNoise
from dataset import DataLoader_Imagenet_val, validation_bsd300, validation_kodak, validation_Set14
from utils import load_pretrained_model, calculate_ssim, calculate_psnr


# 验收标准：Gaussion 25, BSD300: PSNR: 30.79, SSIM:0.873

parser = argparse.ArgumentParser()
parser.add_argument("--noisetype", type=str, default="gauss25")
parser.add_argument('--data_dir', type=str, default='./Imagenet_val')
parser.add_argument('--val_dirs', type=str, default='./validation')
parser.add_argument('--save_model_path', type=str, default='./results')
parser.add_argument('--log_name', type=str, default='unet_gauss25_b4e100r02')
parser.add_argument('--gpu_devices', default='0', type=str)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--n_feature', type=int, default=48)
parser.add_argument('--n_channel', type=int, default=3)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--log_iters', type=int, default=100)
parser.add_argument('--n_snapshot', type=int, default=1)
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--patchsize', type=int, default=256)
parser.add_argument("--Lambda1", type=float, default=1.0)
parser.add_argument("--Lambda2", type=float, default=1.0)
parser.add_argument("--increase_ratio", type=float, default=2.0)

opt, _ = parser.parse_known_args()
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
operation_seed_counter = 0
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices


class AugmentNoise(object):
    def __init__(self, style):
        print(style)
        if style.startswith('gauss'):
            self.params = [
                float(p) / 255.0 for p in style.replace('gauss', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "gauss_fix"
            elif len(self.params) == 2:
                self.style = "gauss_range"
        elif style.startswith('poisson'):
            self.params = [
                float(p) for p in style.replace('poisson', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "poisson_fix"
            elif len(self.params) == 2:
                self.style = "poisson_range"

    def add_train_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            std = std * paddle.ones(shape)
            global operation_seed_counter
            operation_seed_counter += 1
            paddle.seed(operation_seed_counter)
            noise = paddle.normal(mean=0.0, std=std)
            return x + noise

    def add_valid_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)


def checkpoint(net, epoch, name):
    save_model_path = os.path.join(opt.save_model_path, opt.log_name)
    os.makedirs(save_model_path, exist_ok=True)
    model_name = 'epoch_{}_{:03d}.pdparams'.format(name, epoch)
    save_model_path = os.path.join(save_model_path, model_name)
    paddle.save(net.state_dict(), save_model_path)
    print('Checkpoint saved to {}'.format(save_model_path))


def space_to_depth(x, block_size):
    n, c, h, w = x.shape
    unfolded_x = paddle.nn.functional.unfold(x, block_size, strides=block_size)
    return unfolded_x.reshape([n, c * block_size**2, h // block_size,
                           w // block_size])


def generate_mask_pair(img):
    # prepare masks (N x C x H/2 x W/2)
    n, c, h, w = img.shape

    mask1 = paddle.zeros(shape=(n * h // 2 * w // 2 * 4, ),
                        dtype='int64')
    mask2 = paddle.zeros(shape=(n * h // 2 * w // 2 * 4, ),
                        dtype='int64')
    # prepare random mask pairs

    idx_pair = paddle.to_tensor(
        np.array([[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]]).astype('int64'))
    global operation_seed_counter
    operation_seed_counter += 1
    paddle.seed(operation_seed_counter)
    rd_idx = paddle.randint(low=0,
                            high=8,
                            shape=(n * h // 2 * w // 2,))
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += paddle.arange(start=0,
                                end=n * h // 2 * w // 2 * 4,
                                step=4,
                                dtype='int64').reshape([-1, 1])

    # get masks
    rd_pair_idx1 = rd_pair_idx[:, 0]
    rd_pair_idx2 = rd_pair_idx[:, 1]    
    updates = paddle.ones(rd_pair_idx1.shape,dtype='int64')
    mask1 = paddle.scatter(mask1, rd_pair_idx1, updates, overwrite=True)
    updates = paddle.ones(rd_pair_idx2.shape,dtype='int64')
    mask2 = paddle.scatter(mask2, rd_pair_idx2, updates, overwrite=True)
    mask1 = paddle.cast(mask1, 'bool')
    mask2 = paddle.cast(mask2, 'bool')
    
    # mask1[rd_pair_idx1] = 1
    # mask2[rd_pair_idx2] = 1
    return mask1, mask2


def generate_subimages(img, mask):
    n, c, h, w = img.shape
    subimage = paddle.zeros((n,
                           c,
                           h // 2,
                           w // 2),
                           dtype=img.dtype)
    # per channel
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = paddle.transpose(img_per_channel, [0, 2, 3, 1]).reshape([-1])
        subimage[:, i:i + 1, :, :] = paddle.transpose(img_per_channel[mask].reshape([
            n, h // 2, w // 2, 1]),[0, 3, 1, 2])
    return subimage


# Training Set
TrainingDataset = DataLoader_Imagenet_val(opt.data_dir, patch=opt.patchsize)
TrainingLoader = DataLoader(dataset=TrainingDataset,
                            num_workers=8,
                            batch_size=opt.batchsize,
                            shuffle=True,
                            drop_last=True)

# Validation Set
Kodak_dir = os.path.join(opt.val_dirs, "Kodak")
BSD300_dir = os.path.join(opt.val_dirs, "BSD300")
Set14_dir = os.path.join(opt.val_dirs, "Set14")
# valid_dict = {
#     "Kodak": validation_kodak(Kodak_dir),
#     "BSD300": validation_bsd300(BSD300_dir),
#     "Set14": validation_Set14(Set14_dir)
# }
valid_dict = {
    "BSD300": validation_bsd300(BSD300_dir)
}

# Noise adder
noise_adder = AugmentNoise(style=opt.noisetype)

# Network
network = UNet(in_nc=opt.n_channel,
               out_nc=opt.n_channel,
               n_feature=opt.n_feature)
# about training scheme
num_epoch = opt.n_epoch
ratio = num_epoch / 100
lr = paddle.optimizer.lr.MultiStepDecay(learning_rate=opt.lr, milestones=[
                                         int(20 * ratio) - 1,
                                         int(40 * ratio) - 1,
                                         int(60 * ratio) - 1,
                                         int(80 * ratio) - 1
                                     ],
                                        gamma=opt.gamma
                                        )
optimizer = paddle.optimizer.Adam(parameters=network.parameters(), learning_rate=lr)

print("Batchsize={}, number of epoch={}".format(opt.batchsize, opt.n_epoch))

checkpoint(network, 0, "model")
print('init finish')

for epoch in range(1, opt.n_epoch + 1):
    cnt = 0

    current_lr = lr.get_lr()
    print("LearningRate of Epoch {} = {}".format(epoch, current_lr))

    network.train()
    for iteration, data in enumerate(TrainingLoader):
        st = time.time()
        clean = data[0] / 255.0
        noisy = noise_adder.add_train_noise(clean)

        mask1, mask2 = generate_mask_pair(noisy)
        noisy_sub1 = generate_subimages(noisy, mask1)
        noisy_sub2 = generate_subimages(noisy, mask2)
        with paddle.no_grad():
            noisy_denoised = network(noisy)
        noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1)
        noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2)

        noisy_output = network(noisy_sub1)
        noisy_target = noisy_sub2
        Lambda = epoch / opt.n_epoch * opt.increase_ratio
        diff = noisy_output - noisy_target
        exp_diff = noisy_sub1_denoised - noisy_sub2_denoised

        loss1 = paddle.mean(diff**2)
        loss2 = Lambda * paddle.mean((diff - exp_diff)**2)
        loss_all = opt.Lambda1 * loss1 + opt.Lambda2 * loss2
        loss_all.backward()
        optimizer.step()
        network.clear_gradients()
        if iteration % opt.log_iters == 0:
            print(
                '{:04d} {:05d} Loss1={:.6f}, Lambda={}, Loss2={:.6f}, Loss_Full={:.6f}, Time={:.4f}'
                .format(epoch, iteration, np.mean(loss1.item()), Lambda,
                        np.mean(loss2.item()), np.mean(loss_all.item()),
                        time.time() - st))

    lr.step()

    if epoch % opt.n_snapshot == 0 or epoch == opt.n_epoch:
        network.eval()
        # save checkpoint
        checkpoint(network, epoch, "model")
        # validation
        save_model_path = os.path.join(opt.save_model_path, opt.log_name,
                                       systime)
        validation_path = os.path.join(save_model_path, "validation")
        os.makedirs(validation_path, exist_ok=True)
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
                    noisy_im = noise_adder.add_valid_noise(im)
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

                    # visualization
                    if i == 0 and idx < 5:
                        save_path = os.path.join(
                            validation_path,
                            "{}_{:03d}-{:03d}_clean.png".format(
                                valid_name, idx, epoch))
                        Image.fromarray(origin255).convert('RGB').save(
                            save_path)
                        save_path = os.path.join(
                            validation_path,
                            "{}_{:03d}-{:03d}_noisy.png".format(
                                valid_name, idx, epoch))
                        Image.fromarray(noisy255).convert('RGB').save(
                            save_path)
                    if i == 0 and idx < 5:
                        save_path = os.path.join(
                            validation_path,
                            "{}_{:03d}-{:03d}_denoised.png".format(
                                valid_name, idx, epoch))
                        Image.fromarray(pred255).convert('RGB').save(save_path)

            psnr_result = np.array(psnr_result)
            avg_psnr = np.mean(psnr_result)
            avg_ssim = np.mean(ssim_result)
            print(f"[EVAL] {valid_name}: psnr:{avg_psnr} ssim:{avg_ssim}")
            log_path = os.path.join(validation_path,
                                    "A_log_{}.csv".format(valid_name))
            with open(log_path, "a") as f:
                f.writelines("{},{},{}\n".format(epoch, avg_psnr, avg_ssim))
