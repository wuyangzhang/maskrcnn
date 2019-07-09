import argparse
import os
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.prediction import test


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )


    parser.add_argument(
        "--dataset",
        default='',
        metavar='FILE',
        type=str,
    )

    parser.add_argument(
        "--model",
        default='',
        metavar='FILE',
        type=str,
    )

    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    test(cfg,args.dataset,args.model)


if __name__ == "__main__":
    main()
