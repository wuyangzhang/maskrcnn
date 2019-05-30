import argparse
import os
from maskrcnn_benchmark.modeling.prediction import train
from maskrcnn_benchmark.config import cfg


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )


    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    train(cfg)


if __name__ == "__main__":
    main()
