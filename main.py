import os
import argparse

from data_loader import get_loader
from solver import Solver
from torch.backends import cudnn


def main(config):
    cudnn.benchmark = True

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)

    data_loader = get_loader(
        config.image_dir,
        config.image_size,
        config.batch_size,
        config.dataset,
        config.mode,
        config.num_workers,
        config.wmat_dir,
    )
    data_loader_test = get_loader(
        config.image_val_dir,
        config.image_size,
        config.batch_size,
        config.dataset,
        config.mode,
        config.num_workers,
        config.wmat_dir,
    )

    solver = Solver(data_loader, data_loader_test, config)
    if config.mode == "train_mask":
        solver.train_mask()
    elif config.mode == "eval_mask":
        solver.eval_mask()
    elif config.mode == "test_accuracy":
        solver.test_accuracy()
    elif config.mode == "test_embedding":
        solver.test_embedding()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_size", type=int, default=128, help="host image size")
    parser.add_argument("--num_channels", type=int, default=64, help="channels for discriminator")
    parser.add_argument(
        "--dataset",
        type=str,
        default="train_mask",
        choices=["train_mask", "eval_mask", "test_accuracy", "test_embedding"],
    )
    parser.add_argument("--batch_size", type=int, default=16, help="mini-batch size")
    parser.add_argument("--lambda1", type=float, default=3, help="the weights of message loss")
    parser.add_argument("--lambda2", type=float, default=1, help="the weights of image loss")
    parser.add_argument("--lambda3", type=float, default=0.001, help="the weights of GAN loss")
    parser.add_argument("--num_epoch", type=int, default=100, help="number of total epochs")
    parser.add_argument("--embedding_epoch", type=int, default=0, help="The adversarial training epoch")
    parser.add_argument(
        "--distortion",
        type=str,
        default="ScreenShootingMB",
        choices=["Identity", "ScreenShooting", "ScreenShootingMB"],
    )
    parser.add_argument("--test_iters", type=int, default=99, help="test model from this step")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--mode",
        type=str,
        default="train_mask",
        choices=["train_mask", "eval_mask", "test_accuracy", "test_embedding"],
    )
    parser.add_argument("--weight_path", type=str, default="", help="评估时直接指定 .pth")
    parser.add_argument("--image_dir", type=str, default="Datasets/COCOMask/train/train_class/")
    parser.add_argument("--image_val_dir", type=str, default="Datasets/COCOMask/val/val_class/")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--model_save_dir", type=str, default="models")
    parser.add_argument("--model_name", type=str, default="Encoder_Decoder_Model")
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--wmat_dir", type=str, default="results/WatermarkMatrix/w.mat")
    parser.add_argument("--log_step", type=int, default=40)
    parser.add_argument("--model_save_step", type=int, default=1)

    config = parser.parse_args()
    print(config)
    main(config)
