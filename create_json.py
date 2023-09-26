import argparse

from utils.general import create_data_lists

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dirs", type=str, nargs="+", default='')
    parser.add_argument("--val_dirs", type=str, nargs="+", default='')
    parser.add_argument("--shape", type=int, default=96)

    opt = parser.parse_args()
    create_data_lists(train_folders=opt.train_dirs,
                      test_folders=opt.val_dirs,
                      min_size=opt.shape)
