from .solution import Solution
from brick_id.dataset.catalog import Brick
import matplotlib.pyplot as plt
import os
import cv2
import csv
import numpy as np
from typing import Tuple, List
from brick_id.dataset.catalog import allowable_parts

class MatchableBrick:
    def __init__(self, brick_str: str, characteristic_images: List[str]):
        max_sift_features = 50  # How many descriptors should we try to find in an images?
        self.sift = cv2.SIFT.create(max_sift_features)
        cwd = os.path.dirname(os.path.abspath(__file__))
        b200c_dir = os.path.join(cwd, '../../B200C/64/')
        self.brick_dir = os.path.join(b200c_dir, brick_str)
        self.brick_id = Brick['_' + brick_str]

        self.descriptors = self.get_descriptors(characteristic_images)

    def get_descriptors(self, img_names: str) -> np.ndarray:
        descriptors = []
        for img_name in img_names:
            img_path = os.path.join(self.brick_dir, img_name + '.jpg')
            image = cv2.imread(img_path)
            kp, des = self.sift.detectAndCompute(image, None)
            descriptors.append(des)
        return descriptors

    def try_match(self, matcher: cv2.BFMatcher, des: np.ndarray) -> Tuple[Brick, int]:
        n_matches = 0
        for descriptor in self.descriptors:
            n_matches = max(n_matches, self.get_matches(matcher, descriptor, des))
            n_matches = max(n_matches, self.get_matches(matcher, des, descriptor))
        return self.brick_id, n_matches

    def get_matches(self, matcher: cv2.BFMatcher, des1: np.ndarray, des2: np.ndarray) -> int:
        # Matching test from the OpenCV tutorial:
        # https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
        matches = matcher.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good_matches = 0
        try:
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches += 1
            return good_matches
        except ValueError:
            print(f'Trouble retrieving matches! Problematic brick is {self.brick_id}')
            return 0


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


        self.max_sift_features = 50 # How many descriptors should we try to find in an images?
        self.min_descriptor_count = 8 # At least how many descriptors need to be found to call it a good image?
        self.sift = cv2.SIFT.create(self.max_sift_features)
        self.em = cv2.ml.EM.create()
        self.matcher = cv2.BFMatcher()

        self.matchable_bricks = [
            MatchableBrick('62462', ['577', '578', '814', '1065']),
            MatchableBrick('11211', ['110', '1367', '428', '149']),
            MatchableBrick('87552', ['109', '1029', '619', '431']),
            MatchableBrick('3749', ['4', '142', '1023', '565']),
            MatchableBrick('11214', ['24', '35', '141', '325', '410']),
            MatchableBrick('11090', ['2','13','15','16','26','49','68']),
            MatchableBrick('11212', ['3', '11', '26','31','34','39','57']),
            MatchableBrick('18677', ['23','57','58','68','70','89']),
            MatchableBrick('87620', ['22', '78', '115', '142', '182', '367', '384']),
            MatchableBrick('2454', ['269', '279', '334', '444', '647']),
            MatchableBrick('85984', ['24', '45', '144', '228', '256', '357']),
            MatchableBrick('60474', ['0', '10', '16', '21', '37', '184']),
            MatchableBrick('11458', ['2', '17', '66', '69', '70', '105']),
            MatchableBrick('18654', ['12', '16', '22', '43', '61']),
            MatchableBrick('3020', ['64', '68', '78', '106', '145', '312']),
            MatchableBrick('6558', ['18', '99', '105', '142', '494']),
            MatchableBrick('2429', ['22', '23', '28', '73', '74']),
            MatchableBrick('41677', ['138', '152', '173', '196', '242']),
            MatchableBrick('32140', ['17', '24', '184', '318', '485']),
            MatchableBrick('60479', ['324', '361', '3880']),
            MatchableBrick('3008', ['191', '315', '396']),
            MatchableBrick('3023', ['15', '57', '58', '107', '108']),
            MatchableBrick('32523', ['23', '29', '115', '116', '117']),
            MatchableBrick('2654', ['3', '8', '11', '20', '21']),
            MatchableBrick('48336', ['7', '16', '17', '25', '28', '35']),
            MatchableBrick('87083', ['148', '225', '275', '283', '311']),
            MatchableBrick('44728', ['22', '30', '31', '32', '226']),
            MatchableBrick('4274', ['15', '25', '46', '63', '73']),
            MatchableBrick('2357', ['15', '18', '34', '33', '26', '32']),
            MatchableBrick('85861', ['2', '6', '13', '17', '104', '136']),
            MatchableBrick('3622', ['16', '17', '8', '33', '34', '39']),
            MatchableBrick('60478', ['15', '16', '18', '34', '100']),
            MatchableBrick('3040', ['105', '110', '144', '160', '194']),
            MatchableBrick('88072', ['15', '22', '75', '130', '158']),
            MatchableBrick('2430', ['5', '22', '23', '34', '57', '63']),
            MatchableBrick('3795', ['15', '17', '64', '102', '159']),
            MatchableBrick('4286', ['58', '66', '150', '185']),
            MatchableBrick('87580', ['2', '3', '10', '17', '19']),
            MatchableBrick('3710', ['16', '24', '111', '109']),

            # MatchableBrick('3039', []),
            # MatchableBrick('41770', []),
            # MatchableBrick('4085', []),
            # MatchableBrick('3032', []),
            # MatchableBrick('3666', []),
            # MatchableBrick('41769', []),
            # MatchableBrick('85080', []),
            # MatchableBrick('3070b', []),
            # MatchableBrick('30136', []),
            # MatchableBrick('14704', []),
            # MatchableBrick('3705', []),
            # MatchableBrick('30374', []),
            # MatchableBrick('3460', []),
            # MatchableBrick('41740', []),
            # MatchableBrick('3713', []),
            # MatchableBrick('15379', []),
            # MatchableBrick('3034', []),
            # MatchableBrick('32952', []),
            # MatchableBrick('15573', []),
            # MatchableBrick('3004', []),
            #
            # MatchableBrick('53451', []),
            # MatchableBrick('4589', []),
            # MatchableBrick('2420', []),
            # MatchableBrick('99207', []),
            # MatchableBrick('32000', []),
            # MatchableBrick('3673', []),
            # MatchableBrick('60601', []),
            # MatchableBrick('24201', []),
            # MatchableBrick('61252', []),
            # MatchableBrick('64644', []),
            # MatchableBrick('87994', []),
            # MatchableBrick('4519', []),
            # MatchableBrick('4081b', []),
            # MatchableBrick('3958', []),
            # MatchableBrick('3037', []),
            # MatchableBrick('3021', []),
            # MatchableBrick('32278', []),
            # MatchableBrick('4070', []),
            # MatchableBrick('47457', []),
        ]

        self.catalog = allowable_parts()

        # self.required_bricks = [
        #
        # model_filename = 'chuck_em_model.xml'
        # model_path = os.path.join(cwd, model_filename)
        #
        # # # Do we have the model already?
        # # if not os.path.exists(model_path):
        # #     print('Model not found, looking for saved training data...')
        # #     # Okay, if we don't, do we at least have the training data available?
        # #     training_data_filename = 'chuck_training_data.csv'
        # #     training_data_path = os.path.join(cwd, training_data_filename)
        # #
        # #     feature_array = np.array(0)
        # #     if not os.path.exists(training_data_path):
        # #         print('Training data not found, creating training data. This will take a while...')
        # #         # Create the training data. This is going to take some time.
        #
        # # How many images are we going to read?
        # n_images = 0
        #
        # # Search the directories first so we can pre-allocate the feature array
        # # Path scraping from https://stackoverflow.com/a/59938961/5171120
        # brick_ids = [f.name for f in os.scandir(b200c_dir) if f.is_dir()]
        # for brick_id in brick_ids:
        #     if brick_id not in self.required_bricks:
        #         continue
        #     brick_path = os.path.join(b200c_dir, brick_id)
        #     image_paths = [f.path for f in os.scandir(brick_path) if f.is_file()]
        #     n_images += len(image_paths)
        # print(f'Need to load {n_images} images')
        # sift_descriptor_length = 128
        #
        # # This answer on Stack Overflow was helpful for understanding how to format the descriptors to be consumed
        # # by the TrainData.create method: https://stackoverflow.com/a/53730463/5171120
        # # One row for each image, columns for the SIFT descriptors, plus one more column for the label
        # feature_array = np.zeros((n_images, self.max_sift_features*sift_descriptor_length + 1),
        #                          dtype=np.float32)
        # current_ct = 0
        # notify_ct = 50
        # most_descriptors = 0
        # descriptor_dict = dict()
        # descriptor_max_count = 0
        #
        # # Create a BFMatcher object
        # bf = cv2.BFMatcher()
        #
        # for brick_id in brick_ids:
        #     if brick_id not in self.required_bricks:
        #         continue
        #     brick_path = os.path.join(b200c_dir, brick_id)
        #     image_paths = [f.path for f in os.scandir(brick_path) if f.is_file()]
        #     views = list()
        #
        #     for image_path in image_paths:
        #         current_ct += 1
        #         if np.mod(current_ct, notify_ct) == 0:
        #             print(f'Currently on {current_ct} of {n_images}')
        #         image = cv2.imread(image_path)
        #         kp, des = self.sift.detectAndCompute(image, None)
        #
        #         # Bad image, didn't find any descriptors!
        #         if des is None:
        #             continue
        #         des = np.array(des)
        #
        #         # Poor image, only found a few descriptors!
        #         if des.shape[0] < self.min_descriptor_count:
        #             continue
        #
        #         if len(views) == 0:
        #             views.append(des)
        #         else:
        #             view_found = False
        #             for view in views:
        #                 # Want to match 50% of the descriptors to call it a consistent view... I think?
        #                 min_matches = int(0.5*des.shape[0])
        #
        #                 # Matching test from the OpenCV tutorial:
        #                 # https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
        #
        #                 matches = bf.knnMatch(des, view, k=2)
        #                 # Apply ratio test
        #                 good_matches = 0
        #                 for m, n in matches:
        #                     if m.distance < 0.75 * n.distance:
        #                         good_matches += 1
        #                 print(f'Found {good_matches} matches')
        #
        #                 # If we meet the min_matches threshold, then we've matched a known view of the object and we
        #                 # can stop looking.
        #                 if good_matches >= min_matches:
        #                     view_found = True
        #                     break
        #
        #             # If we didn't match a known view, maybe it's a new view!
        #             if not view_found:
        #                 views.append(des)
        #                 print(f'Added a view, currently at {len(views)} views')
        #     print(f'Found {len(views)} views, view has {views[0].shape[0]} descriptors')
        #

                # if des.shape[0] > most_descriptors:
                #     most_descriptors = des.shape[0]
                #
                # if des.shape[0] > self.max_sift_features:
                #     des = des[:self.max_sift_features, :]
                # des = des.reshape(1, -1)
                #
                # feature_array[current_ct - 1, 0:des.shape[1]] = des
                # feature_array[current_ct - 1, self.max_sift_features * sift_descriptor_length] = int(
                #     Brick['_' + brick_id])
        #         if most_descriptors > self.max_sift_features:
        #             print(f'Maximum number of descriptors was exceeded; most found was {most_descriptors}')
        #         # Loop  through the directories again, this time extracting features from the images
        #         with open(os.path.join(cwd, 'chuck_training_data.csv'), 'w') as csvfile:
        #             writer = csv.writer(csvfile, delimiter=',')
        #             writer.writerows(feature_array)
        #     else:
        #         print('Found training data, loading now...')
        #         # This section of code from Google's AI Overview in response to the search query "AttributeError: type object
        #         # 'cv2.ml.TrainData' has no attribute 'loadFromCSV'", searched on 30NOV2024
        #
        #         # Load the data from the CSV file
        #         data = np.loadtxt(training_data_path, delimiter=",")
        #
        #         # Split the data into features and labels
        #         features = data[:, :-1].astype(np.float32)
        #         labels = data[:, -1].astype(np.int32)
        #
        #         print('Loaded the CSV, creating a training dataset...')
        #         training_data = cv2.ml.TrainData.create(features, cv2.ml.ROW_SAMPLE, labels)
        #
        #         print('Training the model...')
        #         self.em.train(training_data)
        #
        #         print(f'Done! Writing the model to {model_path}')
        #         fs = cv2.FileStorage(model_path, cv2.FILE_STORAGE_WRITE)
        #         self.em.write(fs)
        #         print('Done!')
        # else:
        #     self.em.load(model_path)



    def identify(self, blob):
        kp, des = self.sift.detectAndCompute(blob, None)
        if des is None:
            print('Failed to extract descriptors from the provided image!')
            return Brick.NOT_IN_CATALOG
        best_match = 0
        best_brick = Brick.NOT_IN_CATALOG
        for brick in self.matchable_bricks:
            test_brick, test_match = brick.try_match(self.matcher, des)
            if test_match > best_match:
                best_match = test_match
                best_brick = test_brick
        brick_id = Brick(best_brick).name
        if best_brick > 0:
            brick_id = brick_id[1:]
            brick_description = self.catalog[brick_id]
        else:
            brick_description = 'NOT IN CATALOG'
        print(f'Guessed {brick_id} for that image; description: {brick_description}')

        return best_brick











































































