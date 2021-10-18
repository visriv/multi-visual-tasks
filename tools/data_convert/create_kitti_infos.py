import argparse
import pickle
from pathlib import Path

from mvt.datasets.data_converter.data_converter import (
    read_dataset_file
)
from mvt.datasets.data_converter.kitti_converter import (
    calculate_num_points_in_gt,
    get_kitti_data_info,
    convert_cam_to_lidar_info
)

def create_kitti_info_file(data_path, save_path=None, relative_path=True):
    """
    create the file of dataset info
    :param data_path: the root path of dataset
    :param save_path: saving path for the file
    :param relative_path: whether using relative path
    :return:
    """
    dataset_folder = Path(__file__).resolve().parent / "kitti_dataset"
    train_ids = read_dataset_file(str(dataset_folder / "train.txt"))
    val_ids = read_dataset_file(str(dataset_folder / "val.txt"))
    test_ids = read_dataset_file(str(dataset_folder / "test.txt"))

    print("Generate info. this may take several minutes.")
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    kitti_infos_train = get_kitti_data_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        image_ids=train_ids,
        relative_path=relative_path)
    calculate_num_points_in_gt(data_path, kitti_infos_train, relative_path)

    filename = save_path / 'kitti_cam_infos_train.pkl'
    print(f"Kitti info train file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    kitti_infos_val = get_kitti_data_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        image_ids=val_ids,
        relative_path=relative_path)
    calculate_num_points_in_gt(data_path, kitti_infos_val, relative_path)

    filename = save_path / 'kitti_cam_infos_val.pkl'
    print(f"Kitti info val file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)

    filename = save_path / 'kitti_cam_infos_trainval.pkl'
    print(f"Kitti info trainval file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)

    kitti_infos_test = get_kitti_data_info(
        data_path,
        training=False,
        label_info=False,
        velodyne=True,
        calib=True,
        image_ids=test_ids,
        relative_path=relative_path)
    filename = save_path / 'kitti_cam_infos_test.pkl'
    print(f"Kitti info test file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)



def create_kitti_lidar_info_file(data_path, save_path=None):
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)

    train_info_path = save_path / 'kitti_cam_infos_train.pkl'
    with open(train_info_path, 'rb') as f:
        train_infos = pickle.load(f)
    train_lidar_infos = []
    for info in train_infos:
        res_info = convert_cam_to_lidar_info(info)
        train_lidar_infos.append(res_info)
    filename = save_path / 'kitti_infos_train.pkl'
    print(f"Kitti info train file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(train_lidar_infos, f)

    val_info_path = save_path / 'kitti_cam_infos_val.pkl'
    with open(val_info_path, 'rb') as f:
        val_infos = pickle.load(f)
    val_lidar_infos = []
    for info in val_infos:
        res_info = convert_cam_to_lidar_info(info)
        val_lidar_infos.append(res_info)
    filename = save_path / 'kitti_infos_val.pkl'
    print(f"Kitti info val file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(val_lidar_infos, f)

    filename = save_path / 'kitti_infos_trainval.pkl'
    print(f"Kitti info trainval file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(train_lidar_infos + val_lidar_infos, f)

    test_info_path = save_path / 'kitti_cam_infos_test.pkl'
    with open(test_info_path, 'rb') as f:
        test_infos = pickle.load(f)
    test_lidar_infos = []
    for info in test_infos:
        res_info = convert_cam_to_lidar_info(info)
        test_lidar_infos.append(res_info)
    filename = save_path / 'kitti_infos_test.pkl'
    print(f"Kitti info test file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(test_lidar_infos, f)


def arg_parser():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
    parser.add_argument(
        '--root-path',
        type=str,
        default='./data/kitti',
        help='specify the root path of dataset')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='./data/kitti',
        help='specify the output path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parser()
    kitti_data_prep(root_path=args.root_path)
