# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import time
from os import path as osp

import cv2
from PIL import Image

import numpy as np
import paddle
from paddle import inference
from paddle.inference import Config, create_predictor
import paddle.nn.functional as F

from utils import calculate_ssim, calculate_psnr


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    # general params
    parser = argparse.ArgumentParser("PaddleVideo Inference model script")
    parser.add_argument("-i", "--input_file", type=str, help="input file path")
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--params_file", type=str)
    parser.add_argument("--save_dir", type=str, default="output/inference_img")

    # params for predict
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=8000)
    parser.add_argument("--enable_benchmark", type=str2bool, default=False)
    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_threads", type=int, default=None)

    return parser.parse_args()


def create_paddle_predictor(args):
    config = Config(args.model_file, args.params_file)
    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()
        if args.cpu_threads:
            config.set_cpu_math_library_num_threads(args.cpu_threads)
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()
            if args.precision == "fp16":
                config.enable_mkldnn_bfloat16()

    # config.disable_glog_info()
    config.switch_ir_optim(args.ir_optim)  # default true
    if args.use_tensorrt:
        # choose precision
        if args.precision == "fp16":
            precision = inference.PrecisionType.Half
        elif args.precision == "int8":
            precision = inference.PrecisionType.Int8
        else:
            precision = inference.PrecisionType.Float32

        # calculate real max batch size during inference when tenrotRT enabled
        num_seg = 1
        num_views = 1
        max_batch_size = args.batch_size * num_views * num_seg
        config.enable_tensorrt_engine(precision_mode=precision,
                                      max_batch_size=max_batch_size)

    config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)

    return config, predictor


def parse_file_paths(input_path: str) -> list:
    if osp.isfile(input_path):
        files = [
            input_path,
        ]
    else:
        files = os.listdir(input_path)
        files = [
            file for file in files
            if (file.endswith(".png"))
        ]
        files = [osp.join(input_path, file) for file in files]
    return files


def postprocess(input_file,input_file_names, output,save_dir, print_output=True):
    """
    output: list
    """
    prediction = np.transpose(output[0], [0, 2, 3, 1])
    prediction = np.clip(prediction, 0, 1)
    pred255 = np.clip(prediction * 255.0 + 0.5, 0,
                      255).astype(np.uint8)
    for i in range(pred255.shape[0]):
        pred255_im = Image.fromarray(pred255[i], "RGB")
        input_file_name = input_file_names[i]
        os.makedirs(save_dir, exist_ok=True)
        pred255_im.save(os.path.join(save_dir, input_file_name + "_denoise.png"))

    origin = np.transpose(input_file[0], [0, 2, 3, 1])
    origin = np.clip(origin, 0, 1)
    origin = np.clip(origin * 255.0 + 0.5, 0,
                      255).astype(np.uint8)

    psnr_list = []
    ssim_list = []

    for i in range(pred255.shape[0]):
        cur_psnr = calculate_psnr(origin[i].astype(np.float32),
                                  pred255[i].astype(np.float32))
        psnr_list.append(cur_psnr)
        cur_ssim = calculate_ssim(origin[i].astype(np.float32),
                                  pred255[i].astype(np.float32))
        ssim_list.append(cur_ssim)
    psnr_result = np.array(psnr_list)
    avg_psnr = np.mean(psnr_result)
    avg_ssim = np.mean(ssim_list)
    if print_output:
        print(f"\tPSNR: {avg_psnr}")
        print(f"\tSSIM: {avg_ssim}")

def main():
    args = parse_args()

    model_name = 'N2N'
    print(f"Inference model({model_name})...")
    # InferenceHelper = build_inference_helper(cfg.INFERENCE)

    inference_config, predictor = create_paddle_predictor(args)

    # get input_tensor and output_tensor
    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    input_tensor_list = []
    output_tensor_list = []
    for item in input_names:
        input_tensor_list.append(predictor.get_input_handle(item))
    for item in output_names:
        output_tensor_list.append(predictor.get_output_handle(item))

    # get the absolute file path(s) to be processed
    files = parse_file_paths(args.input_file)

    if args.enable_benchmark:
        test_video_num = 50
        num_warmup = 0

        # instantiate auto log
        import auto_log
        pid = os.getpid()
        autolog = auto_log.AutoLogger(
            model_name="N2N",
            model_precision=args.precision,
            batch_size=args.batch_size,
            data_shape="dynamic",
            save_path="./output/auto_log.lpg",
            inference_config=inference_config,
            pids=pid,
            process_name=None,
            gpu_ids=0 if args.use_gpu else None,
            time_keys=['preprocess_time', 'inference_time', 'postprocess_time'],
            warmup=num_warmup)

    # Inferencing process
    batch_num = args.batch_size
    for st_idx in range(0, len(files), batch_num):
        ed_idx = min(st_idx + batch_num, len(files))

        # auto log start
        if args.enable_benchmark:
            autolog.times.start()

        # Pre process batched input
        batched_inputs = [files[st_idx:ed_idx]]
        imgs = []
        input_file_names = []
        for inp in batched_inputs[0]:
            img = Image.open(inp)
            img = np.array(img)
            img = cv2.resize(img, (256, 256))
            img = np.array(img, dtype=np.float32) / 255.0
            noisy_im = np.array(img + np.random.normal(size=img.shape) * (25 / 255),
                     dtype=np.float32)
            H = noisy_im.shape[0]
            W = noisy_im.shape[1]
            val_size = (max(H, W) + 31) // 32 * 32
            noisy_im = np.pad(
                noisy_im,
                [[0, val_size - H], [0, val_size - W], [0, 0]],
                'reflect')
            noisy_im = noisy_im.transpose([2, 0, 1])
            noisy_im = noisy_im[np.newaxis, :,:,:]

            imgs.append(noisy_im)
            input_file_names.append(inp.split('/')[-1].split('.')[0])
        imgs = np.concatenate(imgs)
        batched_inputs = [imgs]
        # get pre process time cost
        if args.enable_benchmark:
            autolog.times.stamp()

        # run inference
        for i in range(len(input_tensor_list)):
            input_tensor_list[i].copy_from_cpu(batched_inputs[i])
        predictor.run()

        batched_outputs = []
        for j in range(len(output_tensor_list)):
            batched_outputs.append(output_tensor_list[j].copy_to_cpu())

        # get inference process time cost
        if args.enable_benchmark:
            autolog.times.stamp()

        postprocess(batched_inputs, input_file_names,batched_outputs, args.save_dir, not args.enable_benchmark)

        # get post process time cost
        if args.enable_benchmark:
            autolog.times.end(stamp=True)

        # time.sleep(0.01)  # sleep for T4 GPU

    # report benchmark log if enabled
    if args.enable_benchmark:
        autolog.report()


if __name__ == "__main__":
    main()
