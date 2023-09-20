import sys
import argparse
import os
import os.path
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from modules.train import Train
from modules.assert_model import AssertModel

mode_choices = [
    "train",
    "assert-model",
    "sample"
]

model_types = [
    "cnn_vae"
]

def main():
    parser = argparse.ArgumentParser(description="PyIGenerate: Python implementations for image generators")

    parser.add_argument("--mode", help="Mode to be used", choices=mode_choices, type=str, required=True)
    parser.add_argument("--config-file", help="Config file", type=str)
    parser.add_argument("--img-width", help="Image width", type=int, default=128)
    parser.add_argument("--img-height", help="Image height", type=int, default=128)
    parser.add_argument("--device", help="Device used for training", choices=["cpu", "cuda"], type=str, default="cuda")
    parser.add_argument("--gpu-index", help="GPU index", type=int, default=0)
    parser.add_argument("--input-img-dir", help="Input image directory", type=str)
    parser.add_argument("--model-file", help="Model file", type=str, default="model.pth")
    parser.add_argument("--model-type", help="Model type", type=str, default="cnn-vae")
    parser.add_argument("--batch-size", help="Batch size", type=int, default=1)
    parser.add_argument("--in-channels", help="In Channels", type=int, default=3)
    parser.add_argument("--loss-type", help="Loss type", type=str, default="mse")

    args = parser.parse_args()

    mode            = args.mode
    config_file     = args.config_file
    img_width       = args.img_width
    img_height      = args.img_height
    device          = args.device
    gpu_index       = args.gpu_index
    input_img_dir   = args.input_img_dir
    model_file      = args.model_file
    in_channels     = args.in_channels
    model_type      = args.model_type
    loss_type       = args.loss_type

    if mode == "train":
        params = {
            'img_width':        img_width,
            'img_height':       img_height,
            'device':           device,
            'gpu_index':        gpu_index,
            'input_img_dir':    input_img_dir,
            'model_file':       model_file,
            'in_channels':      in_channels,
            'model_type':       model_type,
            'loss_type':        loss_type
        }

        if config_file:
            with open(config_file) as json_file:
                params = json.load(json_file)

        cmd = Train(params=params)
        cmd.execute()

    elif mode == "assert-model":
        params = {
            'device':       device,
            'gpu_index':    gpu_index,
            'model_type':   model_type,
            'img_width':    img_width,
            'img_height':   img_height,
            'in_channels':  in_channels
        }

        cmd = AssertModel(params=params)
        cmd.execute()

    else:
        print('Invalid mode')

if __name__ == '__main__':
    main()
