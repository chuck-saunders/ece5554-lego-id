import sys
import cv2
import os

from brick_id.solutions.kayla_solution import KaylaSolution
from brick_id.solutions.shane_solution import ShaneSolution
from brick_id.get_blobs import get_blobs
from solutions.chuck_solution import ChuckSolution
#from solutions.kayla_solution import KaylaSolution
#from solutions.shane_solution import ShaneSolution
from brick_id.voter import vote_on
#from dataset.catalog import allowable_parts


def identify(path: str):
    #print(f'Attempting to identify file at {path}')
    img = cv2.imread(path)
    blobs = get_blobs(img)
    solutions = [ChuckSolution(), KaylaSolution(), ShaneSolution()]
    # results = list()
    # current_blob = 0
    # for blob in blobs:
    #     guesses = list()
    #     for solution in solutions:
    #         guess = solution.identify(blob)
    #         guesses.append(guess)
    #     result = vote_on(blob, guesses)
    #     print(f'Blob {current_blob} identified as {result}')
    #     current_blob += 1
    #     results.append((blob, result))
    # # TODO: Show the results, calculate scores, etc.



if __name__ == '__main__':
    file_path = ''
    try:
        file_path = sys.argv[1]
    except IndexError:
        print(f'No arg passed to identify; loading test.png as the default image')
        file_path = '../imgs/dataset_1_light.jpg'
    # Did we get a relative path?
    if not os.path.exists(file_path):
        cwd = os.path.abspath(os.path.dirname(__file__))
        full_path = os.path.join(cwd, file_path)
        if not os.path.exists(full_path):
            raise Exception(f'Could not find provided file "{file_path}" at that location or, assuming it was a '
                            f'relative path, at "{full_path}" either')
        file_path = full_path
    identify(file_path)