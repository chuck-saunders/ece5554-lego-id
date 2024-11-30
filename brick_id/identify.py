import sys
import cv2
import os

from solutions.chuck_solution import ChuckSolution
from brick_id.solutions.kayla_solution import KaylaSolution
from brick_id.solutions.shane_solution import ShaneSolution
from brick_id.object_segmentation import object_segmentation
import matplotlib.pyplot as plt
from brick_id.voter import vote_on


def identify(path: str):
    img = cv2.imread(path)

    object_extents = object_segmentation(img)
    solutions = [ChuckSolution(), KaylaSolution(), ShaneSolution()]
    results = list()
    for object_extent in object_extents:
        xmin, xmax, ymin, ymax = object_extent
        cropped_img = img[ymin:ymax, xmin:xmax]
        plt.imshow(cropped_img)
        plt.axis('off')
        plt.show()
        guesses = list()
        for solution in solutions:
            guess = solution.identify(cropped_img)
            guesses.append(guess)
        print(f'Got {len(guesses)} guesses')
        result = vote_on(cropped_img, guesses)
        results.append((cropped_img, result))
    # TODO: Show the results, calculate scores, etc.


if __name__ == '__main__':
    file_path = ''
    try:
        file_path = sys.argv[1]
    except IndexError:
        print(f'No arg passed to identify; loading test.png as the default image')
        file_path = '../imgs/dataset_1.jpg'
    # Did we get a relative path?
    if not os.path.exists(file_path):
        cwd = os.path.abspath(os.path.dirname(__file__))
        full_path = os.path.join(cwd, file_path)
        if not os.path.exists(full_path):
            raise Exception(f'Could not find provided file "{file_path}" at that location or, assuming it was a '
                            f'relative path, at "{full_path}" either')
        file_path = full_path
    identify(file_path)