import functools
import torch
import numpy as np
import sys

sys.path.append("..")
from inference_api_base import main


def argparse_hook(parser):
    parser.add_argument(
        "--model", default="checkpoint/model_best.ckpt", type=str, help="path to model"
    )


def make_predict_func(args):
    from models import DeepConvolutionalUNet

    net = DeepConvolutionalUNet(hidden_size=512 // 2 + 1)
    net = torch.nn.DataParallel(net)
    checkpoint = torch.load(args.model, map_location="cpu")
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    return functools.partial(predict, net)


def predict(net, batch: list):
    with torch.no_grad():
        return net(torch.from_numpy(np.vstack(batch))).cpu().numpy()


main(
    "denoise",
    make_predict_func,
    argparse_hook,
    input_is_audio=True,
    output_is_audio=True,
    batch_same_len=False,
)
