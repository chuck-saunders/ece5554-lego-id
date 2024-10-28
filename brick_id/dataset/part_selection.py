#import random
import numpy as np
import csv

from brick_id.dataset.catalog import allowable_parts
from brick_id.attributes.capture_attributes import LightingConditions, BackgroundConditions

class RandomizedDataset(object):
    # Seed the random number generator so the sequence is random but repeatable.
    # The class number seems like as good a seed as anything else.
    rng = np.random.default_rng(5554)

    def __init__(self, max_part_count = 100):
        self.max_part_count = max_part_count
        num_lighting_conditions = len(LightingConditions)
        num_background_conditions = len(BackgroundConditions)
        parts_list = list(allowable_parts().items())
        num_available_parts = len(parts_list)
        self.lighting_condition = RandomizedDataset.rng.choice(list(LightingConditions))
        self.background_condition = RandomizedDataset.rng.choice(list(BackgroundConditions))
        self.part_count = RandomizedDataset.rng.integers(1, max_part_count)
        self.selected_part_counts = dict()
        for i in range(self.part_count):
            selected_part = RandomizedDataset.rng.integers(1, num_available_parts)
            selected_part_count = self.selected_part_counts.get(selected_part, 0)
            selected_part_count += 1
            self.selected_part_counts[selected_part] = selected_part_count
        self.selected_parts = list()
        for key, val in self.selected_part_counts.items():
            self.selected_parts.append((parts_list[key], val))


# Parameters per the test procedure described in the project proposal
num_images = 25
datasets = list()
for i in range(num_images):
    datasets.append(RandomizedDataset())

# Put together how many parts need to be retrieved. The min pick list is the minimum required to be pulled, which may
# result in the same physical brick appearing in multiple images. The extended pick list is what's required to have
# unique physical bricks across all images.
min_pick_list = dict()
extended_pick_list = dict()
for dataset in datasets:
    for key, value in dataset.selected_part_counts.items():
        existing_min_count = min_pick_list.get(key, 0)
        existing_extended_count = extended_pick_list.get(key, 0)
        required_count = value
        min_count = max(existing_min_count, required_count)
        extended_count = existing_extended_count + required_count
        min_pick_list[key] = min_count
        extended_pick_list[key] = extended_count

parts_list = list(allowable_parts().items())

with open('pick_list.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Min Qty','Extended Qty','Description','Possible Image URL'])
    for key, value in min_pick_list.items():
        # Reference images can typically be found with the following URL pattern:
        # https://img.bricklink.com/ItemImage/PL/<your part number>.png
        reference_image_url = f'https://img.bricklink.com/ItemImage/PL/{parts_list[key][0]}.png'
        writer.writerow([value, extended_pick_list[key], parts_list[key], reference_image_url])