from .solution import Solution
from brick_id.dataset.catalog import Brick
import matplotlib.pyplot as plt
import os
import cv2
import csv
import numpy as np
from typing import Tuple, List
import pickle
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
        self.descriptors_per_brick = 5000 # How many descriptors should we choose to create a representative collection?
        # I tried using 5000 descriptors per brick and the model crashed when I tried to train it.
        self.sift = cv2.SIFT.create(self.max_sift_features)
        self.matcher = cv2.BFMatcher()
        self.bow_extractor = cv2.BOWImgDescriptorExtractor(self.sift, self.matcher)
        #self.em = cv2.ml.EM.create()
        #self.predictor = cv2.ml.EM.create()

        # Here's the concept: Get all the features,
        self.predictor = cv2.ml.SVM.create()
        self.catalog = allowable_parts()


        model_filename = 'chuck_kmeans_svm_model.dat'
        model_path = os.path.join(cwd, model_filename)
        centers_path = os.path.join(cwd, 'cluster_450_centers.dat')

        # Do we have the model already?
        if not os.path.exists(model_path):
            print('Model not found, looking for saved training data...')
            # Okay, if we don't, do we at least have the training data available?
            training_data_filename = 'chuck_training_data.csv'
            training_data_path = os.path.join(cwd, training_data_filename)

            feature_array = np.array(0)
            if not os.path.exists(training_data_path):
                # I did this once for pulling 5000 descriptors per brick, and it's going to be faster to sample that
                # than to re-extract features to build a new csv.
                dataset_5k_path = os.path.join(cwd, 'chuck_training_data_5k.csv')
                if os.path.exists(dataset_5k_path):
                    # Load the data from the CSV file
                    print(f'Found the 5000-descriptor-per-brick dataset, downsampling to {self.descriptors_per_brick} '
                          f'per brick')
                    data = np.loadtxt(dataset_5k_path, delimiter=",")

                    # Split the data into features and labels
                    full_dataset_features = data[:, :-1].astype(np.float32)
                    full_dataset_labels = data[:, -1].astype(np.int32)
                    rows_per_image = 5000
                    rows_to_skip = int(rows_per_image/self.descriptors_per_brick)
                    full_dataset_features = full_dataset_features[::rows_to_skip, :]
                    n_features = full_dataset_features.shape[0]
                    full_dataset_labels = full_dataset_labels[::rows_to_skip].reshape(n_features,1)
                    print(f'Min label is {np.min(full_dataset_labels)}, max label is {np.max(full_dataset_labels)}')

                    # Write the downsampled dataset to the target CSV
                    print('Finished collecting image features, writing to CSV...')
                    with open(os.path.join(cwd, training_data_filename), 'w') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',')
                        feature_array = np.hstack((full_dataset_features, full_dataset_labels))
                        writer.writerows(feature_array)
                    print('Finished creating the CSV, creating a training dataset...')
                else:
                    print('Training data not found, creating training data. This will take a while...')
                    # Create the training data. This is going to take some time.

                    # How many images are we going to read?
                    n_images = 0

                    # Search the directories first so we can pre-allocate the feature array
                    # Path scraping from https://stackoverflow.com/a/59938961/5171120
                    brick_ids = [f.name for f in os.scandir(b200c_dir) if f.is_dir()]
                    for brick_id in brick_ids:
                        brick_path = os.path.join(b200c_dir, brick_id)
                        image_paths = [f.path for f in os.scandir(brick_path) if f.is_file()]
                        n_images += len(image_paths)
                    print(f'Need to load {n_images} images')

                    sift_descriptor_length = 128

                    # This answer on Stack Overflow was helpful for understanding how to format the descriptors to be
                    # consumed by the TrainData.create method: https://stackoverflow.com/a/53730463/5171120
                    # One row for each image, columns for the SIFT descriptors, plus one more column for the label
                    full_dataset_features = np.zeros((len(brick_ids)*self.descriptors_per_brick, sift_descriptor_length),
                                                 dtype=np.float32)
                    full_dataset_labels = np.zeros((len(brick_ids)*self.descriptors_per_brick, 1), dtype=np.int32)
                    current_brick = 0
                    current_ct = 0
                    notify_ct = 500

                    # Don't keep resizing these arrays, just preallocate a lot and then resize only if necessary. I know
                    # these are huge arrays, but running this once gave 182499 as the peak number of descriptors found
                    # for a single brick (when self.max_sift_features is 50).
                    brick_descriptors = np.zeros((182499, sift_descriptor_length), dtype=np.float32)
                    brick_labels = np.zeros((182499, 1), dtype=np.int32)
                    most_descriptors_found = 0
                    for brick_id in brick_ids:
                        current_array_index = 0
                        brick_path = os.path.join(b200c_dir, brick_id)
                        image_paths = [f.path for f in os.scandir(brick_path) if f.is_file()]

                        brick_int_value = int(Brick['_' + brick_id])
                        descriptors_found = 0
                        for image_path in image_paths:
                            current_ct += 1
                            if np.mod(current_ct, notify_ct) == 0:
                                print(f'Currently on {current_ct} of {n_images}')
                            image = cv2.imread(image_path)
                            kp, image_descriptors = self.sift.detectAndCompute(image, None)

                            # Bad image, didn't find any descriptors!
                            if image_descriptors is None:
                                continue
                            image_descriptors = np.array(image_descriptors)
                            descriptor_count = image_descriptors.shape[0]
                            descriptors_found += descriptor_count
                            image_labels = np.ones((image_descriptors.shape[0], 1), dtype=np.int32) * brick_int_value

                            # Add the descriptors to the collection
                            start_idx = current_array_index
                            current_array_index += descriptor_count
                            end_idx = current_array_index
                            if end_idx > brick_descriptors.shape[0]:
                                brick_descriptors.resize((end_idx, sift_descriptor_length))
                                brick_labels.resize((end_idx, 1))
                            brick_descriptors[start_idx:end_idx, :] = image_descriptors
                            brick_labels[start_idx:end_idx] = image_labels

                        # Cache the peak number of descriptors found for a single brick for curiosity
                        most_descriptors_found = np.max((most_descriptors_found, descriptors_found))

                        # Check that we found a reasonable number of descriptors
                        if descriptors_found < self.descriptors_per_brick:
                            raise Exception(f'Expected to find at least {self.descriptors_per_brick} descriptors but '
                                            f'found {descriptors_found} instead for brick {brick_id}')

                        # Pick self.descriptors_per_brick rows randomly from the collection
                        random_rows = np.random.choice(descriptors_found, size=self.descriptors_per_brick, replace=False)

                        # Add those rows to the output
                        start_idx = current_brick * self.descriptors_per_brick
                        current_brick += 1
                        end_idx = current_brick * self.descriptors_per_brick
                        full_dataset_features[start_idx:end_idx, :] = brick_descriptors[random_rows, :]
                        full_dataset_labels[start_idx:end_idx, :] = brick_labels[random_rows, :]
                    print('Finished collecting image features, writing to CSV...')
                    print(f'(the most descriptors found for a single brick was {most_descriptors_found})')
                    with open(os.path.join(cwd, training_data_filename), 'w') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',')
                        feature_array = np.hstack((full_dataset_features, full_dataset_labels))
                        writer.writerows(feature_array)
                    print('Finished creating the CSV, creating a training dataset...')
            else:
                print('Found training data, loading now...')
                # This section of code from Google's AI Overview in response to the search query "AttributeError: type object
                # 'cv2.ml.TrainData' has no attribute 'loadFromCSV'", searched on 30NOV2024

                # Load the data from the CSV file
                data = np.loadtxt(training_data_path, delimiter=",")

                # Split the data into features and labels
                full_dataset_features = data[:, :-1].astype(np.float32)
                full_dataset_labels = data[:, -1].astype(np.int32)
                print('Loaded the CSV, creating a training dataset...')

            bow_filename = 'chuck_bow_dataset.csv'
            bow_path = os.path.join(cwd, bow_filename)
            if not os.path.exists(bow_path):
                if not os.path.exists(centers_path):

                    # You can pick more sizes here, but I did it in increments of 50 from 50 to 500 and increments of 250
                    # from 750 to 2500 and the "elbow" happens around 450. For the purposes of recovering the 'good' cluster
                    # values, we can just use 450 as the cluster size.
                    cluster_sizes = np.array([450])
                    sum_sqr_distances = np.zeros_like(cluster_sizes)
                    current_cluster = 0
                    for cluster_size in cluster_sizes:
                        print(f'Trying to fit kmeans to data with cluster size {cluster_size}...')
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                        sum_sqr_dist, labels, centers = cv2.kmeans(full_dataset_features, cluster_size, None, criteria,10, cv2.KMEANS_RANDOM_CENTERS)
                        sum_sqr_distances[current_cluster] = sum_sqr_dist
                        current_cluster += 1
                        with open(os.path.join(cwd, 'cluster_' + str(cluster_size) + '_labels.dat'), 'wb') as f:
                            pickle.dump(labels, f)
                        with open(os.path.join(cwd, 'cluster_' + str(cluster_size) + '_centers.dat'), 'wb') as f:
                            pickle.dump(centers, f)
                        with open(os.path.join(cwd, 'cluster_' + str(cluster_size) + '_sum_sqr_dist.dat'), 'wb') as f:
                            pickle.dump(sum_sqr_dist, f)
                        print(f'SSD value is {sum_sqr_dist} for {cluster_size} clusters')
                    plt.figure(figsize=(6, 6))
                    plt.plot(cluster_sizes, sum_sqr_distances, '-o')
                    plt.xlabel(r'Number of clusters *k*')
                    plt.ylabel('Sum of squared distance')
                    plt.show()
                else:
                    with open(centers_path, 'rb') as f:
                        centers = pickle.load(f)

                # From Google's AI Overview in response to the query "train bag of words opencv" searched 03DEC2024.

                vocabulary = centers

                self.bow_extractor.setVocabulary(vocabulary)

                brick_ids = [f.name for f in os.scandir(b200c_dir) if f.is_dir()]
                n_images = 0
                for brick_id in brick_ids:
                    brick_path = os.path.join(b200c_dir, brick_id)
                    image_paths = [f.path for f in os.scandir(brick_path) if f.is_file()]
                    n_images += len(image_paths)
                print(f'Need to load {n_images} images')

                n_words = len(vocabulary)

                # This answer on Stack Overflow was helpful for understanding how to format the descriptors to be
                # consumed by the TrainData.create method: https://stackoverflow.com/a/53730463/5171120
                # One row for each image, columns for the SIFT descriptors, plus one more column for the label
                bow_descriptors = np.zeros((n_images, n_words), dtype=np.float32)
                bow_labels = np.zeros((n_images, 1), dtype=np.int32)
                current_brick = 0
                current_ct = 0
                notify_ct = 500
                valid_imgs = 0
                for brick_id in brick_ids:
                    brick_path = os.path.join(b200c_dir, brick_id)
                    image_paths = [f.path for f in os.scandir(brick_path) if f.is_file()]

                    brick_int_value = int(Brick['_' + brick_id])
                    for image_path in image_paths:
                        current_ct += 1
                        if np.mod(current_ct, notify_ct) == 0:
                            print(f'Currently on {current_ct} of {n_images}')
                        img = cv2.imread(image_path)
                        kp, image_descriptors = self.sift.detectAndCompute(img, None)
                        if image_descriptors is None:
                            continue
                        bow_descriptor = self.bow_extractor.compute(img, kp)
                        bow_descriptors[valid_imgs, :] = bow_descriptor
                        bow_labels[valid_imgs, :] = brick_int_value
                        valid_imgs += 1
                bow_descriptors.resize((valid_imgs, bow_descriptors.shape[1]))
                bow_labels.resize((valid_imgs, 1))
                with open(bow_path, 'w') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    feature_array = np.hstack((bow_descriptors, bow_labels))
                    writer.writerows(feature_array)
                print('Finished creating the bag-of-words CSV, creating a training dataset...')
            else:
                print('Found bag-of-words data, loading now...')
                # This section of code from Google's AI Overview in response to the search query "AttributeError: type object
                # 'cv2.ml.TrainData' has no attribute 'loadFromCSV'", searched on 30NOV2024

                with open(centers_path, 'rb') as f:
                    centers = pickle.load(f)
                vocabulary = centers
                self.bow_extractor.setVocabulary(vocabulary)

                # Load the data from the CSV file
                data = np.loadtxt(bow_path, delimiter=",")

                # Split the data into features and labels
                bow_descriptors = data[:, :-1].astype(np.float32)
                bow_labels = data[:, -1].astype(np.int32)
                print('Loaded the bag-of-words CSV, creating a training dataset...')

            training_data = cv2.ml.TrainData.create(bow_descriptors, cv2.ml.ROW_SAMPLE, bow_labels)

            print('Training the model...')

            # Only necessary for Expectation Maximization
            #self.predictor.setClustersNumber(num_bricks)
            self.predictor.train(training_data)
            # self.em.train(training_data)

            print(f'Done! Writing the model to {model_path}')
            self.predictor.save(model_path)
            print('Done!')
        else:
            print('Found the model file, loading it now...')

            with open(centers_path, 'rb') as f:
                centers = pickle.load(f)
            vocabulary = centers
            self.bow_extractor.setVocabulary(vocabulary)

            # self.em.load(model_path, )
            #self.predictor = cv2.ml.EM.load(model_path)
            self.predictor = cv2.ml.SVM.load(model_path)
            # with open(model_path, 'rb') as f:
            #     self.predictor = pickle.load(f)
            print('Done!')



    def identify(self, blob):

        kp, des = self.sift.detectAndCompute(blob, None)
        if des is None:
            print('Failed to extract descriptors from the provided image!')
            return Brick.NOT_IN_CATALOG
        bow_descriptor = self.bow_extractor.compute(blob, kp)
        result = self.predictor.predict(bow_descriptor)
        print(f'Result [0] is:\n{result[0]}\nResult [1] is:\n{result[1]}')
        best_guess = int(result[0])

        brick_id = Brick(best_guess).name
        if best_guess > 0:
            brick_id = brick_id[1:]
            brick_description = self.catalog[brick_id]
        else:
            brick_description = 'NOT IN CATALOG'
        print(f'Guessed {brick_id} for that image; description: {brick_description}')

        return Brick.NOT_IN_CATALOG











































































