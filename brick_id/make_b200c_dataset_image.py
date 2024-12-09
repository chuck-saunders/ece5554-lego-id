import os
import numpy as np
import cv2

cwd = os.path.abspath(os.path.dirname(__file__))
b200c_dir = os.path.join(cwd, '../B200C/64/')
brick_ids = [f.name for f in os.scandir(b200c_dir) if f.is_dir()]
output_img_name = 'b200c_sample_dataset'

idxs = [0, 5, 2, 3, 4]
for img_idx in idxs:
    width = 1920
    height = 1080
    ncols = int(np.ceil(1920/64))
    nrows = int(np.ceil(1080/64))
    output_img = np.zeros((nrows*64, ncols*64, 3))

    current_id_idx = 0
    for row in range(nrows):
        for col in range(ncols):
            brick_id = brick_ids[current_id_idx]
            current_id_idx += 1
            if current_id_idx >= len(brick_ids):
                current_id_idx = 0
            brick_path = os.path.join(b200c_dir, brick_id)
            input_img = cv2.imread(os.path.join(brick_path, str(img_idx) + '.jpg'))
            output_img[row*64:(row+1)*64, col*64:(col+1)*64, :] = input_img

    cv2.imwrite(os.path.join(cwd, output_img_name + '_' + str(img_idx) + '.png'), output_img)
