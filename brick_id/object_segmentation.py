import numpy as np
import cv2
from matplotlib import pyplot as plt
from typing import List


def point_in_box(pt: List[float], box: np.array) -> bool:
    """
    Checks if a point (x,y) exists within a given box (xmin, xmax, ymin, ymax)

    :param pt: The given (x,y) point to check
    :param box: The given box (xmin, xmax, ymin, ymax)
    :return: True if the point is inside the box, False otherwise
    """
    x = pt[0]
    y = pt[1]
    x_min, x_max, y_min, y_max = box
    if x_min < x < x_max and y_min < y < y_max:
        return True
    return False


def boxes_intersect(box1: np.array, box2: np.array) -> bool:
    """
    Checks if two boxes intersect
    :param box1: A box in (xmin, xmax, ymin, ymax) format
    :param box2: A box in (xmin, xmax, ymin, ymax) format
    :return: True if at least one point from one box exists inside the other box, False otherwise
    """
    return box1_has_corner_inside_box2(box1, box2) or box1_has_corner_inside_box2(box2, box1)


def box1_has_corner_inside_box2(box1: np.array, box2: np.array) -> bool:
    """
    Checks if one box has a corner inside another box
    :param box1: A box in (xmin, xmax, ymin, ymax) format
    :param box2: A box in (xmin, xmax, ymin, ymax) format
    :return: True if one box has a corner inside the other box, False otherwise
    """
    x1_min, x1_max, y1_min, y1_max = box1

    return (point_in_box([x1_min, y1_min], box2) or point_in_box([x1_min, y1_max], box2) or
            point_in_box([x1_max, y1_min], box2) or point_in_box([x1_max, y1_max], box2))


def merge_boxes(box1: np.array, box2: np.array) -> np.array:
    """
    Merges two boxes, taking minimums and maximums of the x and y values for both boxes
    :param box1: A box in (xmin, xmax, ymin, ymax) format
    :param box2: A box in (xmin, xmax, ymin, ymax) format
    :return: The smallest box that contains both boxes; min(box1, box2) to max(box1, box2)
    """
    x1_min, x1_max, y1_min, y1_max = box1
    x2_min, x2_max, y2_min, y2_max = box2

    x_min = np.min([x1_min, x2_min, x1_max, x2_max])
    x_max = np.max([x1_min, x2_min, x1_max, x2_max])
    y_min = np.min([y1_min, y2_min, y1_max, y2_max])
    y_max = np.max([y1_min, y2_min, y1_max, y2_max])
    return np.array([x_min, x_max, y_min, y_max])


def collapse_bounding_boxes_with_padding(boxes: List[np.array], img: np.array, padding: int = 0,
                                         draw_debug_image: bool = False) -> bool:
    """
    Iterates through a list of bounding boxes in the form (xmin, xmax, ymin, ymax), and checks for any overlapping or
    fully inscribed boxes. An optional padding value is applied for the purposes of determining this check; the
    overlap value helps to merge individual features into a complete object.
    :param boxes: A list of numpy arrays in the form [xmin, xmax, ymin, ymax]
    :param img: The image in which the bounding boxes are drawn (used for debugging purposes)
    :param padding: The amount of padding to apply when checking for overlap, in pixels
    :param draw_debug_image: Whether to draw debug steps; this will show the bounding boxes that are being compared.
    :return: True if at least one pair of bounding boxes has been merged.
    """
    box1_idx = 0
    debug_thickness = 3
    boxes_altered = False
    while box1_idx < len(boxes):
        box1 = boxes[box1_idx]
        box2_idx = box1_idx + 1
        while box2_idx < len(boxes):
            box2 = boxes[box2_idx]

            # Pad the outer box to consume close-but-not-quite-touching boxes.
            padded_box1 = [box1[0]-padding, box1[1]+padding, box1[2]-padding, box1[3]+padding]
            if boxes_intersect(padded_box1, box2):
                boxes_altered = True

                # The logic here took me a minute. First, merge the boxes.
                box3 = merge_boxes(box1, box2)

                # Replace the first box with the merged box
                boxes[box1_idx] = box3
                box1 = boxes[box1_idx]

                # Delete the consumed box
                del boxes[box2_idx]

                # Reset the box2 index; this restarts the search with the new bigger bounding box.
                box2_idx = box1_idx + 1

                if draw_debug_image:
                    debug_img = img.copy()
                    cv2.rectangle(debug_img, (box1[0], box1[2]), (box1[1], box1[3]), (255, 0, 0), debug_thickness)
                    cv2.rectangle(debug_img, (box2[0], box2[2]), (box2[1], box2[3]), (255, 0, 0), debug_thickness)
                    cv2.rectangle(debug_img, (box3[0], box3[2]), (box3[1], box3[3]), (0, 255, 255), debug_thickness)
                    debug_img = cv2.resize(debug_img, (1024, 768))
                    cv2.imshow('Collapse Bounding Boxes Debug Image (intersecting)', debug_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                if draw_debug_image:
                    debug_img = img.copy()
                    cv2.rectangle(debug_img, (box1[0], box1[2]), (box1[1], box1[3]), (0, 255, 0), debug_thickness)
                    cv2.rectangle(debug_img, (box2[0], box2[2]), (box2[1], box2[3]), (0, 255, 0), debug_thickness)
                    debug_img = cv2.resize(debug_img, (1024, 768))
                    cv2.imshow('Collapse Bounding Boxes Debug Image (not intersecting)', debug_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                box2_idx += 1
        box1_idx += 1
    return boxes_altered


def object_segmentation(input_img: np.array) -> List[np.array]:
    """
    Segments an image to find individual bricks. This works well when the bricks are spaced such that no brick exists
    within the rectangular extents of any other brick, when the background is uniform, and when there is some color
    difference between the brick and the background.
    :param input_img: The input image to be segmented
    :return: A list of bounding boxes that outlines each object in the scene
    """
    # Don't overwrite the original with contours, bounding boxes, etc.
    img = input_img.copy()

    # Canny edge detection to try to identify the brick edges
    edges = cv2.Canny(img, 15, 85)

    # Find contours from the Canny edges
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sometimes noise gets picked up as an object, so only keep the edges that have at lease min_pts points in the
    # contour
    min_pts = 25
    big_contours = []
    bounding_boxes = []
    for contour in contours:
        if contour.shape[0] > min_pts:
            big_contours.append(contour)

            # Get the min/max values to create a bounding box from the contour
            xs = np.squeeze(contour[:, :, 0])
            ys = np.squeeze(contour[:, :, 1])
            x_min, x_max = np.min(xs), np.max(xs)
            y_min, y_max = np.min(ys), np.max(ys)
            bounding_boxes.append([x_min, x_max, y_min, y_max])
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)

    # Some sub-features (studs, anti-studs, etc.) get detected, and sometimes the entire circumference of a brick
    # isn't detected as a singular feature. It's possible to collapse the detected bounding boxes by checking for
    # vertices that exist inside of other bounding boxes.
    collapse_bounding_boxes_with_padding(bounding_boxes, img, padding=50, draw_debug_image=False)

    for bounding_box in bounding_boxes:
        cv2.rectangle(img, (bounding_box[0], bounding_box[2]), (bounding_box[1], bounding_box[3]), (255, 0, 0), 1)

    # Draw contours on the original image
    cv2.drawContours(img, big_contours, -1, (0, 255, 0), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    return bounding_boxes