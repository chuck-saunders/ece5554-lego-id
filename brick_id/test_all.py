import sys
import cv2
import os

from brick_id.dataset.catalog import Brick
from solutions.chuck_solution import ChuckSolution
from brick_id.solutions.kayla_solution import KaylaSolution
from brick_id.solutions.shane_solution import ShaneSolution
from brick_id.object_segmentation import object_segmentation
import matplotlib.pyplot as plt
import csv
from brick_id.voter import vote_on
from glob import glob


def test_all():
    # Test against all re-recaptured images:
    cwd = os.path.abspath(os.path.dirname(__file__))
    dataset_ids = range(10)

    # Pull ground truth for those images:
    ground_truth_data = dict()
    with open(os.path.join(cwd, '../imgs/rerecaptured_ground_truth.csv'), 'r') as f:
        csv_data = csv.reader(f)
        for row in csv_data:
            x_center = int(row[1])
            y_center = int(row[2])
            brick_id = Brick['_' + row[3]]
            gt_list = ground_truth_data.get(int(row[0]), [])
            gt_list.append([x_center, y_center, brick_id])
            ground_truth_data[int(row[0])] = gt_list

    solutions = [ChuckSolution()]  # , KaylaSolution(), ShaneSolution()]

    results = dict()
    for dataset_id in dataset_ids:
        result = list()

        ground_truth = ground_truth_data[dataset_id]

        img_path = os.path.join(cwd, '../imgs/rerecaptured_dataset_' + str(dataset_id) + '.jpg')
        img = cv2.imread(img_path)
        object_extents = object_segmentation(img)

        for object_extent in object_extents:
            padding = 0
            xmin, xmax, ymin, ymax = object_extent
            cropped_img = img[(ymin-padding):(ymax+padding), (xmin-padding):(xmax+padding)]
            guesses = list()
            for solution in solutions:
                guess = solution.identify(cropped_img)
                guesses.append([type(solution), guess])
                print(f'{type(solution)} guessed {guess}')

            # For the Brick Enum, 0 = NOT_IN_CATALOG. This is the default if we don't find the corresponding entry in
            # our ground truth collection, which happens if the object_segmentation picks up a shadow or dirt, etc.
            ground_truth_label = ['Ground Truth', 0]
            for entry in ground_truth:
                x_center = entry[0]
                y_center = entry[1]
                if xmin <= x_center <= xmax and ymin <= y_center <= ymax:
                    ground_truth_label = ['Ground Truth', entry[2]]
            print(f'Ground truth is actually: {ground_truth_label[1]}')
            result = [ground_truth_label, guesses]
        results[dataset_id] = result
    with open(os.path.join(cwd, 'results.csv'), 'w', newline='') as csvfile:
        for dataset_id in results.keys():
            result = results[dataset_id]
            row = list()
            row.append(result[0][0])
            row.append(str(result[0][1]))
            for guess in result[1]:
                row.append(str(guess))
            csvfile.write(row)


if __name__ == '__main__':
    test_all()