import argparse

from mvt.utils.config_util import get_task_cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Print the whole config')
    parser.add_argument('config', help='config file path')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = get_task_cfg(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    print(f'Config:\n{cfg.pretty_text}')


if __name__ == '__main__':
    main()
