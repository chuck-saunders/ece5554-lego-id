import numpy as np
import csv
import requests
import os

from fpdf import FPDF
from PIL import Image

from brick_id.dataset.catalog import allowable_parts, alternative_part_numbers
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


# Trying to get text formatted in the PDF was a bear; apparently there's no way to control the line spacing on a word-
# wrapped cell using fpdf2 so eventually I gave up and wrote this function to manually word-wrap.
def wrap_text_to_column_width(text:str, width: int) -> list[str]:
    if len(text) <= width:
        return [text]
    split_idx = width-1
    if text[split_idx] != ' ':
        while split_idx >= 0:
            if text[split_idx] == ' ':
                break
            split_idx -= 1
    if split_idx < 0:
        raise Exception(f'Failed to find a space that allows formatting text {text} to column width {width}')
    # We're on a space now, so keep everything up to that space:
    result = [text[:split_idx]]
    # Recursively call to split the rest:
    remainder = wrap_text_to_column_width(text[split_idx+1:], width)
    # Join what we've got:
    for item in remainder:
        result.append(item)
    # Return the result
    return result


# The following code based on Google's Generative AI in response to the search "create a table with images and save to
# pdf python" on 30 October 2024.

class PDF(FPDF):
    def imagex(self, image_path, x, y, w=0, h=0):
        if w == 0 and h == 0:
            img = Image.open(image_path)
            w, h = img.size
        self.image(image_path, x, y, w, h)

def export_to_pdf(dataset_id, dataset: RandomizedDataset):
    cwd = os.path.abspath(os.path.dirname(__file__))
    imgs_path = cwd + '/imgs/'
    pdf = PDF()
    pdf.add_page()

    # How big the pdf.get_y() value can get before we should add a page break
    page_height = 260

    # Set cell dimensions
    img_width = 15
    img_height = 15
    description_width = 25
    text_width = 10
    text_height = 5

    # Add table rows
    pdf.set_font("Helvetica", "", 8)
    # Fill color for alternating lines:
    pdf.set_fill_color(225, 225, 225)
    should_fill = False
    lines = 0
    for (part_number, description), count in dataset.selected_parts:
        lines += 1
        should_fill = lines % 2 == 1
        starting_x = pdf.get_x()
        starting_y = pdf.get_y()
        # Okay, how TALL is this row? We know that:
        # 1. Image height will be img_height,
        # 2. Quantities will be broken into four rows, one for the dataset and one for the quantity, so 2*text_height per
        #    test quantity, and then I want two tests stacked per row (because testing shows I can't fit all datasets
        #    horizontally on one page), so 2*(2*text_height) total
        # 3. The description. This uses the wrap_text_to_column_width method defined above, which uses description_width as
        #    the width. This means we need to call the function, see how many lines there are, then multiply that by the
        #    text_height.
        #
        # then take the max of all three above, and that's the row height.
        description_rows = wrap_text_to_column_width(description, 15)
        row_height = max([img_height, (2 * text_height), len(description_rows * text_height)])

        # Do we need to add a new page?
        if starting_y + row_height > page_height:
            pdf.add_page()
            starting_x = pdf.get_x()
            starting_y = pdf.get_y()

        # Add the image
        img_path = imgs_path + part_number + '.png'
        pdf.imagex(img_path, starting_x, starting_y, img_width, img_height)

        # Adding the image only adds an image, it doesn't make a cell or affect the PDF document position. I think having
        # the border helps readability of the document quite a bit, so I'll draw a box around the image that's padded out
        # to the maximum height
        #
        # Don't fill the cell; the background gray doesn't look good with the white background for most cells.
        pdf.cell(img_width, row_height, '', 1, fill=False)

        current_x = starting_x + img_width

        # Kinda similar to the image, I want the description to be printed in multiple lines without borders, so it looks
        # like one cell, but then I get no borders. So, add one large cell with a border!
        pdf.set_xy(current_x, starting_y)

        pdf.cell(description_width, row_height, '', 1, fill=should_fill)
        pdf.set_xy(current_x, starting_y)

        # Print the part description
        for i in range(len(description_rows)):
            line = description_rows[i]
            pdf.set_y(starting_y + i * text_height)
            pdf.set_x(current_x)
            pdf.cell(description_width, text_height, line, fill=False)
        current_x += description_width

        # Print min and ideal quantities
        pdf.set_xy(current_x, starting_y)
        pdf.cell(text_width, row_height, '', 1, fill=should_fill)
        pdf.set_xy(current_x, starting_y)
        pdf.cell(text_width, text_height, 'Need', fill=should_fill)
        pdf.set_xy(current_x, starting_y + text_height)
        pdf.cell(text_width, text_height, str(count), fill=should_fill)
        # Go over the cell one more time to get the border to look nice
        pdf.set_xy(current_x, starting_y)
        pdf.cell(text_width, row_height, '', 1, fill=False)

        if lines == 1:
            pdf.set_xy(current_x + text_width, starting_y)
            lighting_text = 'Lighting condition: ' + str(dataset.lighting_condition)
            background_text = 'Background condition ' + str(dataset.background_condition)
            pdf.cell(5*text_width, text_height, lighting_text)
            pdf.set_xy(current_x + text_width, starting_y + text_height)
            pdf.cell(5*text_width, row_height, background_text)

        # Advance!
        pdf.set_x(starting_x)
        pdf.set_y(starting_y + row_height)

    pdf.output(cwd + '/dataset_' + str(dataset_id) + '.pdf')

cwd = os.path.abspath(os.path.dirname(__file__))
imgs_path = cwd + '/imgs/'

# Parameters per the test procedure described in the project proposal
num_images = 25
datasets = list()
for i in range(num_images):
    datasets.append(RandomizedDataset())
    export_to_pdf(i, datasets[i])

# Put together how many parts need to be retrieved. The min_qty is the minimum required to be pulled, which may result
# in the same physical brick appearing in multiple images. The ideal_qty list is what's required to have unique physical
# bricks across all test images. The scenario_pull_sheet will have the brick identification along with the min/ideal
# quantities and how many are needed for eah
min_qty = dict()
ideal_qty = dict()
scenario_pull_sheet = dict()
parts = allowable_parts()
parts_list = list(parts.items())
for i in range(len(datasets)):
    dataset = datasets[i]
    for key, value in dataset.selected_part_counts.items():
        brick_id = parts_list[key][0]
        existing_min_count = min_qty.get(brick_id, 0)
        existing_extended_count = ideal_qty.get(brick_id, 0)
        required_count = value
        min_count = max(existing_min_count, required_count)
        extended_count = existing_extended_count + required_count
        min_qty[brick_id] = min_count
        ideal_qty[brick_id] = extended_count
        pull_sheet = scenario_pull_sheet.get(brick_id, {})
        pull_sheet[str(i)] = required_count
        # Grab the per-dataset requirements
        scenario_pull_sheet[brick_id] = pull_sheet

# Also grab the per-part min/ideal quantities
for key in scenario_pull_sheet.keys():
    scenario_pull_sheet[key]['min'] = min_qty[key]
    scenario_pull_sheet[key]['ideal'] = ideal_qty[key]


# Need to pull images to make the PDF; the visual reference helps because the official text description isn't clear
# (what is a 'Slope, Curved 2 x 1 x 2/3 Inverted', really?)
alternatives = alternative_part_numbers()
img_idx = 0
if not os.path.exists(imgs_path):
    os.mkdir(imgs_path)

for part_number in scenario_pull_sheet.keys():
    img_idx += 1
    # For this next bit, I want to keep the same part numbers as what is listed in the B200C dataset, but some of the
    # part numbers used in that dataset are the less-common equivalent of a more-common part number. When pulling
    # images, the more-common part numbers are the only parts that image entries.
    #
    # Keep the part number we're TRYING to use, but if we've logged an alternative part number (which is really the more
    # common way to refer to that part) then use the alternative for image downloading.
    image_part_number = part_number
    if part_number in alternatives:
        image_part_number = alternatives[part_number]

    # If we don't already have the image then download it, but if we have it then we don't need to pull it again.
    if not os.path.exists(imgs_path + part_number + '.png'):
        print(f'Downloading image {img_idx} of {len(scenario_pull_sheet)}')
        image_url = 'http://img.bricklink.com/ItemImage/PL/' + image_part_number + '.png'
        # Without including headers you get 403 Forbidden when trying to pull the images; the suggestion to use the
        # headers arg and the content for the headers var both come from Google's Generative AI in response to the
        # search "python requests.get returns 403 forbidden" on 02 Nov 2024.
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
        img_data = requests.get(image_url, headers=headers).content
        with open(imgs_path + part_number + '.png', 'wb') as f:
            f.write(img_data)



# Now we've got the per-scenario, minimum, and ideal quantities for each part as well as a written description and
# representative graphic for each part. Now make a PDF that we can print off and hold while pulling through Lego bins

pdf = PDF()
pdf.add_page()

# How big the pdf.get_y() value can get before we should add a page break
page_height = 260

# Set cell dimensions
img_width = 15
img_height = 15
description_width = 25
text_width = 10
text_height = 5

# Add table rows
pdf.set_font("Helvetica", "", 8)
# Fill color for alternating lines:
pdf.set_fill_color(225, 225, 225)
should_fill = False
lines = 0
for part_number, dataset_requirements in scenario_pull_sheet.items():
    lines += 1
    should_fill = lines%2 == 1
    starting_x = pdf.get_x()
    starting_y = pdf.get_y()
    # Okay, how TALL is this row? We know that:
    # 1. Image height will be img_height,
    # 2. Quantities will be broken into four rows, one for the dataset and one for the quantity, so 2*text_height per
    #    test quantity, and then I want two tests stacked per row (because testing shows I can't fit all datasets
    #    horizontally on one page), so 2*(2*text_height) total
    # 3. The description. This uses the wrap_text_to_column_width method defined above, which uses description_width as
    #    the width. This means we need to call the function, see how many lines there are, then multiply that by the
    #    text_height.
    #
    # then take the max of all three above, and that's the row height.
    description = parts[part_number]
    description_rows = wrap_text_to_column_width(description, 15)
    row_height = max([img_height, 2*(2*text_height), len(description_rows*text_height)])

    # Do we need to add a new page?
    if starting_y + row_height > page_height:
        pdf.add_page()
        starting_x = pdf.get_x()
        starting_y = pdf.get_y()

    # Add the image
    img_path = imgs_path + part_number + '.png'
    pdf.imagex(img_path, starting_x, starting_y, img_width, img_height)

    # Adding the image only adds an image, it doesn't make a cell or affect the PDF document position. I think having
    # the border helps readability of the document quite a bit, so I'll draw a box around the image that's padded out
    # to the maximum height
    #
    # Don't fill the cell; the background gray doesn't look good with the white background for most cells.
    pdf.cell(img_width, row_height, '', 1, fill=False)

    current_x = starting_x + img_width

    # Kinda similar to the image, I want the description to be printed in multiple lines without borders, so it looks
    # like one cell, but then I get no borders. So, add one large cell with a border!
    pdf.set_xy(current_x, starting_y)

    pdf.cell(description_width, row_height, '', 1, fill=should_fill)
    pdf.set_xy(current_x, starting_y)

    # Print the part description
    for i in range(len(description_rows)):
        line = description_rows[i]
        pdf.set_y(starting_y + i*text_height)
        pdf.set_x(current_x)
        pdf.cell(description_width, text_height, line, fill=False)
    current_x += description_width



    # Print min and ideal quantities
    min_qty = dataset_requirements.pop('min')
    ideal_qty = dataset_requirements.pop('ideal')

    pdf.set_xy(current_x, starting_y)
    pdf.cell(text_width, text_height, 'Min', fill=should_fill)
    pdf.set_xy(current_x, starting_y + text_height)
    pdf.cell(text_width, text_height, str(min_qty), fill=should_fill)
    pdf.set_xy(current_x, starting_y)
    # Don't fill the strictly border cells to avoid overwriting the text
    pdf.cell(text_width, 2 * text_height, '', 1, fill=False)

    pdf.set_xy(current_x, starting_y + 2*text_height)
    pdf.cell(text_width, text_height, 'Ideal', fill=should_fill)
    pdf.set_xy(current_x, starting_y + 3*text_height)
    pdf.cell(text_width, text_height, str(ideal_qty), fill=should_fill)
    pdf.set_xy(current_x, starting_y + 2*text_height)
    # Don't fill the strictly border cells to avoid overwriting the text
    pdf.cell(text_width, 2*text_height, '', 1, fill=False)
    current_x += text_width

    for i in range(num_images):
        row_one_offset = 0
        row_two_offset = 1
        if i%2 != 0:
            row_one_offset = 2
            row_two_offset = 3
        qty = dataset_requirements.get(str(i), 0)

        pdf.set_xy(current_x, starting_y + row_one_offset*text_height)
        pdf.cell(text_width, text_height, f'Set {i+1}', fill=should_fill)
        pdf.set_xy(current_x, starting_y + row_two_offset*text_height)
        pdf.cell(text_width, text_height, str(qty), fill=should_fill)

        # Box both lines
        pdf.set_xy(current_x, starting_y + row_one_offset*text_height)
        # Don't fill the strictly border cells to avoid overwriting the text
        pdf.cell(text_width, 2*text_height, '', 1, fill=False)

        if i%2 != 0:
            current_x += text_width

    # Aesthetic issue here; we have an odd number of images to make, which leaves the right/bottom cell missing. Add
    # an empty cell in that location just to keep the grid consistent.
    if num_images%2 != 0:
        pdf.set_xy(current_x, starting_y + 2*text_height)
        pdf.cell(text_width, 2*text_height, '', 1, fill=should_fill)

    # Advance!
    pdf.set_x(starting_x)
    pdf.set_y(starting_y + row_height)
    #pdf.ln()
    # if lines%lines_per_page == 0:
    #     pdf.add_page()


pdf.output(cwd + "/scenario_pull_sheet.pdf")


# with open('pick_list.csv', 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['Min Qty','Ideal Qty','Description','Possible Image URL'])
#     for key, value in min_qty.items():
#         # Reference images can typically be found with the following URL pattern:
#         # https://img.bricklink.com/ItemImage/PL/<your part number>.png
#         reference_image_url = f'https://img.bricklink.com/ItemImage/PL/{parts_list[key][0]}.png'
#         writer.writerow([value, ideal_qty[key], parts_list[key], reference_image_url])
