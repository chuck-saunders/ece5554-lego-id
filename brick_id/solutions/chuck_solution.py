from .solution import Solution
from brick_id.dataset.catalog import Brick
import matplotlib.pyplot as plt
import os
import cv2
import csv
import numpy as np

class ChuckSolution(Solution):
    def __init__(self, **kwargs):
        super().__init__()
        cwd = os.path.dirname(os.path.abspath(__file__))
        b200c_dir = os.path.join(cwd, '../../B200C/64/')
        if not os.path.isdir(b200c_dir):
            print('Expected to find the B200C dataset inside the project root, like ece5554-lego-id/B200C. If you are '
                  'using a Linux environment then you can symlink the dataset to this path; use:')
            print('ln -s /path/to/B200C /path/to/ece5554-lego-id/B200C ')
            print('B200C not found at the expected path!')
            raise FileNotFoundError

        # How many images are we going to read?
        n_images = 0

        # Path scraping from https://stackoverflow.com/a/59938961/5171120
        brick_ids = [f.path for f in os.scandir(b200c_dir) if f.is_dir()]
        for brick_id in brick_ids:
            brick_path = os.path.join(b200c_dir, brick_id)
            image_paths = [f.path for f in os.scandir(brick_path) if f.is_file()]
            n_images += len(image_paths)
        print(f'Need to load {n_images} images')

        max_sift_features = 25
        sift_descriptor_length = 128

        #test_feature_array = np.zeros((n_images, max_sift_features*sift_descriptor_length + 1), dtype=np.float32)

        sift = cv2.SIFT.create(max_sift_features)

        brick_id = '11211'
        test_dir = os.path.join(b200c_dir, brick_id)

        image_paths = [f.path for f in os.scandir(test_dir) if f.is_file()]
        num_images = len(image_paths)
        update_ct = 50
        current_ct = 0
        model_filename = 'chuck_em_model.xml'
        model_path = os.path.join(cwd, model_filename)

        # Do we have the model already?
        # if not os.path.exists(model_path):
        #     # Okay, if we don't, do we at least have the training data available?
        #     training_data_filename = 'chuck_training_data.csv'
        #     training_data_path = os.path.join(cwd, training_data_filename)
        #
        #     if not os.path.exists(training_data_path):
                # Create the training data. This is going to take some time.
                # with open(os.path.join(cwd, 'chuck_training_data.csv'), 'w') as csvfile:
                #     writer = csv.writer(csvfile, delimiter=',')
        # This answer on Stack Overflow was helpful for understanding how to format the descriptors to be consumed
        # by the TrainData.create method: https://stackoverflow.com/a/53730463/5171120
        # One row for each image, columns for the SIFT descriptors, plus one more column for the label
        feature_matrix = np.zeros((5, max_sift_features*sift_descriptor_length + 1), dtype=np.float32)
        for image_path in image_paths:
            current_ct += 1
            if current_ct > 5:
                break
            if np.mod(current_ct, update_ct) == 0:
                print(f'Currently on file {current_ct} of {num_images}...')

            image = cv2.imread(image_path)
            kp, des = sift.detectAndCompute(image, None)
            des = np.array(des).reshape(1,-1)

            feature_matrix[current_ct-1, 0:des.shape[1]] = des
            feature_matrix[current_ct-1, max_sift_features*sift_descriptor_length] = int(Brick['_' + brick_id])
            # writer.writerow([features, int(Brick['_' + brick_id])])

        # This section of code from Google's AI Overview in response to the search query "AttributeError: type object
        # 'cv2.ml.TrainData' has no attribute 'loadFromCSV'", searched on 30NOV2024

        # # Load the data from the CSV file
        # data = np.loadtxt(training_data_path, delimiter=",")
        #
        # # Split the data into features and labels
        # features = data[:, :-1]
        # labels = data[:, -1]

        training_data = cv2.ml.TrainData.create(feature_matrix[:, :-1], cv2.ml.ROW_SAMPLE, feature_matrix[:,-1])

        em = cv2.ml.EM.create()
        em.train(training_data)

        fs = cv2.FileStorage(os.path.join(cwd, "model.xml"), cv2.FILE_STORAGE_WRITE)
        em.write(fs)



    def identify(self, blob):
        # TODO: Write your implementation
        return Brick.NOT_IN_CATALOG


# Need the following parts to be identifiable, at a minimum. These are the parts that are selected for the
# scenario_pull_sheet, which is the complete list of all pieces used in the dataset images.
# 62462
# 11211
# 87552
# 3749
# 11214
# 11090
# 11212
# 18677
# 87620
# 2454
# 85984
# 60474
# 11458
# 18654
# 3020
# 6558
# 2429
# 41677
# 32140
# 60479
# 3008
# 3023
# 32523
# 2654
# 48336
# 87083
# 44728
# 4274
# 2357
# 85861
# 3622
# 60478
# 3040
# 88072
# 2430
# 3795
# 4286
# 87580
# 3710
# 3039
# 41770
# 4085
# 3032
# 3666
# 41769
# 85080
# 3070b
# 30136
# 14704
# 3705
# 30374
# 3460
# 41740
# 3713
# 15379
# 3034
# 32952
# 15573
# 3004
# 53451
# 4589
# 2420
# 99207
# 32000
# 3673
# 60601
# 24201
# 61252
# 64644
# 87994
# 4519
# 4081b
# 3958
# 3037
# 3021
# 32278
# 4070
# 47457