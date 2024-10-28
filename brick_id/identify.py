import sys
import cv2
from solutions.chuck_solution import ChuckSolution
#from solutions.kayla_solution import KaylaSolution
#from solutions.shane_solution import ShaneSolution
from dataset.catalog import allowable_parts


def identify(path: str):
    #print(f'Attempting to identify file at {path}')
    img = cv2.imread(path)
    #solutions = [ChuckSolution()]
    parts_list = list(allowable_parts().items())
    print(f'Parts[0] is {parts_list[0]} and parts[3] is {parts_list[3]}')

if __name__ == '__main__':
    file_path = ''
    try:
        file_path = sys.argv[1]
    except IndexError:
        print(f'No arg passed to identify; loading test.png as the default image')
        file_path = '../imgs/test.png'
    identify(file_path)