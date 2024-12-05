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
            brick_id = row[3]
            gt_list = ground_truth_data.get(int(row[0]), [])
            gt_list.append([x_center, y_center, brick_id])
            ground_truth_data[int(row[0])] = gt_list

    solutions = [KaylaSolution()]# , ChuckSolution()] #, ShaneSolution()]
    yvals = [92758703448.6855,
            86252990707.9004,
            82473136732.5381,
            79875824027.6313,
            77334220285.6919,
            75262491223.2461,
            74254491627.6929,
            73910137355.9356,
            72358133811.8684,
            71943691850.7441,
            68250219853.1689,
            65838127451.4873,
            64898891249.5388,
            63375167433.502,
            62278589416.6235,
            61346186076.6475,
            60422843309.0671,
            59884322711.332]
    xvals = [50,
            100,
            150,
            200,
            250,
            300,
            350,
            400,
            450,
            500,
            750,
            1000,
            1250,
            1500,
            1750,
            2000,
            2250,
            2500]
    plt.plot(xvals, yvals, label='k-means clusters')
    plt.title('k-means clusters')
    plt.xlabel('cluster count')
    plt.ylabel('center distance')
    plt.show()

    results = dict()
    for dataset_id in dataset_ids:
        result = list()

        ground_truth = ground_truth_data[dataset_id]

        img_path = os.path.join(cwd, '../imgs/rerecaptured_dataset_' + str(dataset_id) + '.jpg')
        img = cv2.imread(img_path)
        object_extents = object_segmentation(img, show_segmented_image=False)

        for object_extent in object_extents:
            padding = 0
            xmin, xmax, ymin, ymax = object_extent
            cropped_img = img[(ymin-padding):(ymax+padding), (xmin-padding):(xmax+padding)]
            guesses = list()
            for solution in solutions:
                # This gets an int from 0 to 200 inclusive (0 is NOT_IN_CATALOG, plus bricks 1 to 200)
                guess = solution.identify(cropped_img)
                # Convert the int to an IntEnum, and get the name for that enum
                guess = Brick(guess).name
                # Strip the leading _
                if guess[0] == '_':
                    guess = guess[1:]
                guesser = type(solution)
                guesses.append([guesser, guess])
                print(f'{guesser} guessed {guess}')

            # For the Brick Enum, 0 = NOT_IN_CATALOG. This is the default if we don't find the corresponding entry in
            # our ground truth collection, which happens if the object_segmentation picks up a shadow or dirt, etc.
            ground_truth_label = ['Ground Truth', Brick(0).name]
            for entry in ground_truth:
                x_center = entry[0]
                y_center = entry[1]
                if xmin <= x_center <= xmax and ymin <= y_center <= ymax:
                    ground_truth_label = ['Ground Truth', entry[2]]
            print(f'Ground truth is actually: {ground_truth_label[1]}')
            plt.imshow(cropped_img)
            plt.axis('off')
            plt.show()

            result.append([ground_truth_label, guesses])
        results[dataset_id] = result
    with open(os.path.join(cwd, 'results.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for result in results.items():
            print(f'Result is {result}')
            for guessed_brick in result[1]:
                print(f'Guessed_brick entry is:\n{guessed_brick}')
                dataset_id = result[0]
                ground_truth = guessed_brick[0]
                guesses = guessed_brick[1]
                row = list()
                row.append('Dataset')
                row.append(str(dataset_id))
                row.append(ground_truth[0])
                row.append(str(ground_truth[1]))
                for guesser, guess in guesses:
                    row.append(str(guesser))
                    row.append(str(guess))
                writer.writerow(row)


if __name__ == '__main__':
    test_all()