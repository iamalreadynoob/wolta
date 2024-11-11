def get_extensions(paths):
    from pathlib import Path

    extensions = {}

    for path in paths:
        fname = Path(path).name

        ext = 'none'
        if fname.__contains__('.'):
            ext = fname.split('.')[-1]

        if ext in extensions.keys():
            extensions[ext] += 1
        else:
            extensions[ext] = 1

    return extensions


def dataset_size_same(paths):
    import cv2

    same = True
    height, width = cv2.imread(paths[0]).shape[:2]

    for i in range(1, len(paths)):
        img_h, img_w = cv2.imread(paths[i]).shape[:2]

        if (img_h != height) or (img_w != width):
            same = False
            break

    return same


def dataset_ratio_same(paths):
    import cv2

    same = True
    height, width = cv2.imread(paths[0]).shape[:2]
    ratio = width / height

    for i in range(1, len(paths)):
        img_h, img_w = cv2.imread(paths[i]).shape[:2]
        img_ratio = img_w / img_h

        if ratio != img_ratio:
            same = False
            break

    return same


def crop(img, path=None, crop_width=256, crop_height=256, get_img=False):
    import cv2

    center_height, center_width = img.shape[0] // 2, img.shape[1] // 2

    x = center_width - crop_width // 2
    y = center_height - crop_height // 2

    img = img[y:y + crop_height, x:x + crop_width]

    if path is not None:
        cv2.imwrite(path, img)
    if get_img is True:
        return img


def fill(img, path=None, fill_width=256, fill_height=256, get_img=False):
    import numpy as np
    import cv2

    white_canvas = np.ones((fill_width, fill_height, 3), dtype=np.uint8) * 255

    x_offset = (fill_width - img.shape[1]) // 2
    y_offset = (fill_height - img.shape[0]) // 2

    white_canvas[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img

    if path is not None:
        cv2.imwrite(path, white_canvas)
    if get_img is True:
        return white_canvas


def make_square(dir_from, dir_to, edge_len=256):
    from glob import glob
    import cv2

    paths = glob('{}/*'.format(dir_from))

    for path in paths:
        img = cv2.imread(path)
        name = path.split('/')[-1]

        if img.shape[0] == edge_len and img.shape[1] == edge_len:
            img = img
        elif img.shape[1] / img.shape[0] == 1:
            img = cv2.resize(img, (edge_len, edge_len))
        elif img.shape[0] >= edge_len and img.shape[1] >= edge_len:
            line = min(img.shape[0], img.shape[1])
            img = crop(img, crop_width=line, crop_height=line, get_img=True)
            img = cv2.resize(img, (edge_len, edge_len))
        elif img.shape[0] < edge_len and img.shape[1] < edge_len:
            img = fill(img, fill_width=edge_len, fill_height=edge_len, get_img=True)
        elif img.shape[0] >= edge_len or img.shape[1] >= edge_len:
            line = min(img.shape[0], img.shape[1])
            img = crop(img, crop_width=line, crop_height=line, get_img=True)
            img = cv2.resize(img, (edge_len, edge_len))

        cv2.imwrite('{}/{}'.format(dir_to, name), img)


def dir_split(src_dir, dest_dir, test_size, val_size):
    from glob import glob
    from shutil import copy
    from random import randint
    from pathlib import Path
    import os

    if test_size + val_size < 1:
        os.makedirs(dest_dir, exist_ok=True)
        os.makedirs('{}/train'.format(dest_dir))
        os.makedirs('{}/test'.format(dest_dir))
        os.makedirs('{}/val'.format(dest_dir))

        d_paths = glob('{}/*'.format(src_dir))

        for d_path in d_paths:
            current_dir = Path(d_path).name

            train_dir = '{}/train/{}'.format(dest_dir, current_dir)
            test_dir = '{}/test/{}'.format(dest_dir, current_dir)
            val_dir = '{}/val/{}'.format(dest_dir, current_dir)

            os.makedirs(train_dir)
            os.makedirs(test_dir)
            os.makedirs(val_dir)

            i_paths = glob('{}/*'.format(d_path))

            test_amount = int(len(i_paths) * test_size)
            val_amount = int(len(i_paths) * val_size)

            for place in ['test', 'val']:
                steps = test_amount
                if place == 'val':
                    steps = val_amount

                for _ in range(steps):
                    index = randint(0, len(i_paths) - 1)
                    copy(i_paths[index], '{}/{}/{}/'.format(dest_dir, place, current_dir))
                    del i_paths[index]

            for i_path in i_paths:
                copy(i_path, '{}/train/{}/'.format(dest_dir, current_dir))


def cls_img_counter(dir_path):
    from glob import glob
    from pathlib import Path

    d_paths = glob('{}/*'.format(dir_path))

    cls_amounts = {}

    if len(d_paths) > 0:
        if len(d_paths) > 1 and Path(d_paths[0]).name in ['train', 'test', 'val'] and Path(d_paths[1]).name in ['train', 'test', 'val']:
            for d_path in d_paths:
                c_paths = glob('{}/*'.format(d_path))

                for c_path in c_paths:
                    c_name = Path(c_path).name
                    amount = len(glob('{}/*'.format(c_path)))

                    if c_name in cls_amounts.keys():
                        cls_amounts[c_name] += amount
                    else:
                        cls_amounts[c_name] = amount
        else:
            for d_path in d_paths:
                c_name = Path(d_path).name
                amount = len(glob('{}/*'.format(d_path)))
                cls_amounts[c_name] = amount

    return cls_amounts
