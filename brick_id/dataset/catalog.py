from collections import OrderedDict

# Turns out, the Lego brick IDs are not strictly numeric; quite a few have a 'b' suffix, which means I can't use an
# integer for the model.
#
# I want this to be an OrderedDict so it gets converted to lists deterministically. A normal dict does not guarantee
# order and so converting to list may get the entries out of order, which would then affect how they're sampled when
# building the datasets.
def allowable_parts() -> OrderedDict[str, str]:
    parts_dict = OrderedDict()

    # Brick model numbers are taken from the folder names in the B200C dataset; brick descriptions based on item search
    # results from Bricklink:
    # https://www.bricklink.com/catalogTree.asp?itemType=P
    #
    # Most items (but not all!) can be reached with the following URL pattern:
    # https://www.bricklink.com/v2/catalog/catalogitem.page?P=<your part number>

    # Reference images can typically be found with the following URL pattern:
    # https://img.bricklink.com/ItemImage/PL/<your part number>.png

    parts_dict['99781'] = 'Bracket 1 x 2 - 1 x 2'
    parts_dict['99780'] = 'Bracket 1 x 2 - 1 x 2 Inverted'
    parts_dict['99563'] = 'Minifigure, Utensil Ingot / Bar'
    parts_dict['99207'] = 'Bracket 1 x 2 - 2 x 2 Inverted'
    parts_dict['99206'] = 'Plate, Modified 2 x 2 x 2/3 with 2 Studs on Side'
    parts_dict['98283'] = 'Brick, Modified 1 x 2 with Masonry Profile'
    parts_dict['98138'] = 'Tile, Round 1 x 1'
    parts_dict['93273'] = 'Slope, Curved 4 x 1 x 2/3 Double'
    parts_dict['92946'] = 'Slope 45 2 x 1 with 2/3 Cutout'
    parts_dict['92280'] = 'Plate, Modified 1 x 2 with Clip with Center Cut on Top'
    parts_dict['88323'] = 'Technic, Link Tread Wide with 2 Pin Holes'
    parts_dict['88072'] = 'Plate, Modified 1 x 2 with Bar Arm Up (Horizontal Arm 5mm)'
    parts_dict['87994'] = 'Bar 3L (Bar Arrow)'
    parts_dict['87620'] = 'Brick, Modified Facet 2 x 2'
    parts_dict['87580'] = 'Plate, Modified 2 x 2 with Groove and 1 Stud in Center (Jumper)'
    parts_dict['87552'] = 'Panel 1 x 2 x 2 with Side Supports - Hollow Studs'
    parts_dict['87087'] = 'Brick, Modified 1 x 1 with Stud on Side'
    parts_dict['87083'] = 'Technic, Axle 4L with Stop'
    parts_dict['87079'] = 'Tile 2 x 4'
    parts_dict['85984'] = 'Slope 30 1 x 2 x 2/3'
    parts_dict['85861'] = 'Plate, Round 1 x 1 with Open Stud'
    parts_dict['85080'] = 'Brick, Round Corner 2 x 2 Macaroni with Stud Notch and Reinforced Underside'
    parts_dict['6636'] = 'Tile 1 x 6'
    parts_dict['6632'] = 'Technic, Liftarm Thin 1 x 3 - Axle Holes'
    parts_dict['6558'] = 'Technic, Pin 3L with Friction Ridges'
    parts_dict['6541'] = 'Technic, Brick 1 x 1 with Hole'
    parts_dict['6536'] = 'Technic, Axle and Pin Connector Perpendicular'
    parts_dict['64644'] = 'Minifigure, Utensil Telescope'
    parts_dict['63965'] = 'Bar 6L with Stop Ring'
    parts_dict['63868'] = 'Plate, Modified 1 x 2 with Clip on End (Horizontal Grip)'
    parts_dict['63864'] = 'Tile 1 x 3'
    parts_dict['62462'] = 'Technic, Pin Connector Round 2L with Slot (Pin Joiner Round)'
    parts_dict['61678'] = 'Slope, Curved 4 x 1'
    parts_dict['61409'] = 'Slope 18 2 x 1 x 2/3 with Grille'
    parts_dict['6134'] = 'Hinge Brick 2 x 2 Top Plate'
    parts_dict['61252'] = 'Plate, Modified 1 x 1 with Open O Clip (Horizontal Grip)'
    parts_dict['6091'] = 'Slope, Curved 2 x 1 x 1 1/3 with Recessed Stud'
    parts_dict['60601'] = 'Glass for Window 1 x 2 x 2 Flat Front'
    parts_dict['60592'] = 'Window 1 x 2 x 2 Flat Front'
    parts_dict['60483'] = 'Technic, Liftarm Thick 1 x 2 - Axle Hole'
    parts_dict['60481'] = 'Slope 65 2 x 1 x 2'
    parts_dict['60479'] = 'Plate 1 x 12'
    parts_dict['60478'] = 'Plate, Modified 1 x 2 with Bar Handle on End'
    parts_dict['60474'] = 'Plate, Round 4 x 4 with Hole'
    parts_dict['60470'] = 'Plate, Modified 1 x 2 with 2 U Clips (Horizontal Grip)'
    parts_dict['59443'] = 'Technic, Axle Connector 2L (Smooth with x Hole + Orientation)'
    parts_dict['54200'] = 'Slope 30 1 x 1 x 2/3'
    parts_dict['53451'] = 'Barb / Claw / Horn / Tooth - Small'
    parts_dict['51739'] = 'Wedge, Plate 2 x 4'
    parts_dict['50950'] = 'Slope, Curved 3 x 1'
    parts_dict['49668'] = 'Plate, Modified 1 x 1 with Tooth Horizontal'
    parts_dict['48729'] = 'Bar 1L with Clip Mechanical Claw (Undetermined Type)'
    parts_dict['4865'] = 'Panel 1 x 2 x 1'
    parts_dict['48336'] = 'Plate, Modified 1 x 2 with Bar Handle on Side - Closed Ends'
    parts_dict['47457'] = 'Slope, Curved 2 x 2 x 2/3 with 2 Studs and Curved Sides'
    parts_dict['4740'] = 'Dish 2 x 2 Inverted (Radar)'
    parts_dict['4599b'] = 'Tap 1 x 1 without Hole in Nozzle End'
    parts_dict['4589'] = 'Cone 1 x 1'
    parts_dict['4519'] = 'Technic, Axle 3L'
    parts_dict['4477'] = 'Plate 1 x 10'
    parts_dict['44728'] = 'Bracket 1 x 2 - 2 x 2'
    parts_dict['43723'] = 'Wedge, Plate 3 x 2 Left'
    parts_dict['43722'] = 'Wedge, Plate 3 x 2 Right'
    parts_dict['43093'] = 'Technic, Axle 1L with Pin with Friction Ridges'
    parts_dict['4286'] = 'Slope 33 3 x 1'
    parts_dict['4274'] = 'Technic, Pin 1/2 without Friction Ridges'
    parts_dict['42003'] = 'Technic, Axle and Pin Connector Perpendicular 3L with 2 Pin Holes'
    parts_dict['41770'] = 'Wedge, Plate 4 x 2 Left'
    parts_dict['41769'] = 'Wedge, Plate 4 x 2 Right'
    parts_dict['41740'] = 'Plate, Modified 1 x 4 with 2 Studs with Groove'
    parts_dict['41677'] = 'Technic, Liftarm Thin 1 x 2 - Axle Holes'
    parts_dict['4162'] = 'Tile 1 x 8'
    parts_dict['4085'] = 'Plate, Modified 1 x 1 with Clip Vertical (Undetermined Type)'
    parts_dict['4081b'] = 'Plate, Modified 1 x 1 with Light Attachment - Thick Ring'
    parts_dict['4073'] = 'Plate, Round 1 x 1'
    parts_dict['4070'] = 'Brick, Modified 1 x 1 with Headlight'
    parts_dict['40490'] = 'Technic, Liftarm Thick 1 x 9'
    parts_dict['4032'] = 'Plate, Round 2 x 2 with Axle Hole'
    parts_dict['3958'] = 'Plate 6 x 6'
    parts_dict['3941'] = 'Brick, Round 2 x 2 with Axle Hole'
    parts_dict['3937'] = 'Hinge Brick 1 x 2 Base'
    parts_dict['3832'] = 'Plate 2 x 10'
    parts_dict['3795'] = 'Plate 2 x 6'
    parts_dict['3749'] = 'Technic, Axle 1L with Pin without Friction Ridges'
    parts_dict['3713'] = 'Technic Bush'
    parts_dict['3710'] = 'Plate 1 x 4'
    parts_dict['3705'] = 'Technic, Axle 4L'
    parts_dict['3701'] = 'Technic, Brick 1 x 4 with Holes'
    parts_dict['3700'] = 'Technic, Brick 1 x 2 with Hole'
    parts_dict['3673'] = 'Technic, Pin without Friction Ridges'
    parts_dict['3666'] = 'Plate 1 x 6'
    parts_dict['3665'] = 'Slope, Inverted 45 2 x 1'
    parts_dict['3660'] = 'Slope, Inverted 45 2 x 2 with Flat Bottom Pin'
    parts_dict['3623'] = 'Plate 1 x 3'
    parts_dict['3622'] = 'Brick 1 x 3'
    parts_dict['35480'] = 'Plate, Round 1 x 2 with Open Studs'
    parts_dict['3460'] = 'Plate 1 x 8'
    parts_dict['34103'] = 'Plate, Modified 1 x 3 with 2 Studs (Double Jumper)'
    parts_dict['33909'] = 'Tile, Modified 2 x 2 with Studs on Edge'
    parts_dict['33291'] = 'Plate, Round 1 x 1 with Flower Edge (4 Knobs / Petals)'
    parts_dict['32952'] = 'Brick, Modified 1 x 1 x 1 2/3 with Studs on Side'
    parts_dict['32607'] = 'Plant Plate, Round 1 x 1 with 3 Leaves'
    parts_dict['32526'] = 'Technic, Liftarm, Modified Bent Thick L-Shape 3 x 5'
    parts_dict['32525'] = 'Technic, Liftarm Thick 1 x 11'
    parts_dict['32524'] = 'Technic, Liftarm Thick 1 x 7'
    parts_dict['32523'] = 'Technic, Liftarm Thick 1 x 3'
    parts_dict['3245c'] = 'Brick 1 x 2 x 2 with Inside Stud Holder'
    parts_dict['32316'] = 'Technic, Liftarm Thick 1 x 5'
    parts_dict['32278'] = 'Technic, Liftarm Thick 1 x 15'
    parts_dict['32184'] = 'Technic, Axle and Pin Connector Perpendicular 3L with Center Pin Hole'
    parts_dict['32140'] = 'Technic, Liftarm, Modified Bent Thick L-Shape 2 x 4'
    parts_dict['32123'] = 'Technic Bush 1/2 Smooth'
    parts_dict['32073'] = 'Technic, Axle 5L'
    parts_dict['32064'] = 'Technic, Brick 1 x 2 with Axle Hole'
    parts_dict['32062'] = 'Technic, Axle 2L Notched'
    parts_dict['32054'] = 'Technic, Pin 3L with Friction Ridges and Stop Bush'
    parts_dict['32028'] = 'Plate, Modified 1 x 2 with Door Rail'
    parts_dict['32013'] = 'Technic, Axle and Pin Connector Angled #1'
    parts_dict['32000'] = 'Technic, Brick 1 x 2 with Holes'
    parts_dict['3070b'] = 'Tile 1 x 1'
    parts_dict['3069b'] = 'Tile 1 x 2'
    parts_dict['3068b'] = 'Tile 2 x 2'
    parts_dict['3065'] = 'Brick 1 x 2 without Bottom Tube'
    parts_dict['3062b'] = 'Brick, Round 1 x 1'
    parts_dict['30414'] = 'Brick, Modified 1 x 4 with Studs on Side'
    parts_dict['30413'] = 'Panel 1 x 4 x 1'
    parts_dict['3040'] = 'Slope 45 2 x 1'
    parts_dict['3039'] = 'Slope 45 2 x 2'
    parts_dict['30374'] = 'Bar 4L (Lightsaber Blade / Wand)'
    parts_dict['3037'] = 'Slope 45 2 x 4'
    parts_dict['3035'] = 'Plate 4 x 8'
    parts_dict['3034'] = 'Plate 2 x 8'
    parts_dict['3032'] = 'Plate 4 x 6'
    parts_dict['3031'] = 'Plate 4 x 4'
    parts_dict['3024'] = 'Plate 1 x 1'
    parts_dict['3023'] = 'Plate 1 x 2'
    parts_dict['3022'] = 'Plate 2 x 2'
    parts_dict['3021'] = 'Plate 2 x 3'
    parts_dict['3020'] = 'Plate 2 x 4'
    parts_dict['30136'] = 'Brick, Modified 1 x 2 with Log Profile'
    parts_dict['3010'] = 'Brick 1 x 4'
    parts_dict['3009'] = 'Brick 1 x 6'
    parts_dict['3008'] = 'Brick 1 x 8'
    parts_dict['3005'] = 'Brick 1 x 1'
    parts_dict['3004'] = 'Brick 1 x 2'
    parts_dict['3003'] = 'Brick 2 x 2'
    parts_dict['3002'] = 'Brick 2 x 3'
    parts_dict['3001'] = 'Brick 2 x 4'
    parts_dict['2877'] = 'Brick, Modified 1 x 2 with Grille / Fluted Profile'
    parts_dict['28192'] = 'Slope 45 2 x 1 with Cutout without Stud'
    parts_dict['27925'] = 'Tile, Round Corner 2 x 2 Macaroni'
    parts_dict['2780'] = 'Technic, Pin with Short Friction Ridges'
    parts_dict['26604'] = 'Brick, Modified 1 x 1 with Studs on 2 Sides, Adjacent'
    parts_dict['26603'] = 'Tile 2 x 3'
    parts_dict['26601'] = 'Wedge, Plate 2 x 2 Cut Corner'
    parts_dict['2654'] = 'Plate, Round 2 x 2 with Rounded Bottom (Boat Stud)'
    parts_dict['26047'] = 'Plate, Round 1 x 1 with Bar Handle'
    parts_dict['2540'] = 'Plate, Modified 1 x 2 with Bar Handle on Side - Free Ends'
    parts_dict['25269'] = 'Tile, Round 1 x 1 Quarter'
    parts_dict['24866'] = 'Plate, Round 1 x 1 with Flower Edge (5 Petals)'
    parts_dict['2456'] = 'Brick 2 x 6'
    parts_dict['2454'] = 'Brick 1 x 2 x 5'
    parts_dict['2450'] = 'Wedge, Plate 3 x 3 Cut Corner'
    parts_dict['2445'] = 'Plate 2 x 12'
    parts_dict['2436'] = 'Bracket 1 x 2 - 1 x 4'
    parts_dict['2432'] = 'Tile, Modified 1 x 2 with Bar Handle'
    parts_dict['2431'] = 'Tile 1 x 4'
    parts_dict['2430'] = 'Hinge Plate 1 x 4 Swivel Top'
    parts_dict['2429'] = 'Hinge Plate 1 x 4 Swivel Base'
    parts_dict['24246'] = 'Tile, Round 1 x 1 Half Circle Extended'
    parts_dict['24201'] = 'Slope, Curved 2 x 1 x 2/3 Inverted'
    parts_dict['2420'] = 'Plate 2 x 2 Corner'
    parts_dict['2412b'] = 'Tile, Modified 1 x 2 Grille with Bottom Groove'
    parts_dict['2357'] = 'Brick 2 x 2 Corner'
    parts_dict['22885'] = 'Brick, Modified 1 x 2 x 1 2/3 with Studs on Side'
    parts_dict['22388'] = 'Slope 45 1 x 1 x 2/3 Quadruple Convex Pyramid'
    parts_dict['20482'] = 'Tile, Round 1 x 1 with Bar and Pin Holder'
    parts_dict['18677'] = 'Plate, Modified 1 x 2 with Pin Hole on Bottom'
    parts_dict['18674'] = 'Tile, Round 2 x 2 with Open Stud'
    parts_dict['18654'] = 'Technic, Liftarm Thick 1 x 1 (Spacer)'
    parts_dict['18651'] = 'Technic, Axle 2L with Pin with Friction Ridges'
    parts_dict['15712'] = 'Tile, Modified 1 x 1 with Open O Clip'
    parts_dict['15573'] = 'Plate, Modified 1 x 2 with 1 Stud with Groove and Bottom Stud Holder (Jumper)'
    parts_dict['15535'] = 'Tile, Round 2 x 2 with Hole'
    parts_dict['15392'] = 'Projectile Launcher Part, Trigger for Gun, Mini Blaster / Shooter'
    parts_dict['15379'] = 'Technic, Link Tread'
    parts_dict['15100'] = 'Technic, Pin with Friction Ridges and Pin Hole'
    parts_dict['15070'] = 'Plate, Modified 1 x 1 with Tooth Vertical'
    parts_dict['15068'] = 'Slope, Curved 2 x 2 x 2/3'
    parts_dict['14769'] = 'Tile, Round 2 x 2 with Bottom Stud Holder'
    parts_dict['14719'] = 'Tile 2 x 2 Corner'
    parts_dict['14704'] = 'Plate, Modified 1 x 2 with Small Tow Ball Socket on Side'
    parts_dict['11477'] = 'Slope, Curved 2 x 1 x 2/3'
    parts_dict['11476'] = 'Plate, Modified 1 x 2 with Clip on Side (Horizontal Grip)'
    parts_dict['11458'] = 'Plate, Modified 1 x 2 with Pin Hole on Top'
    parts_dict['11214'] = 'Technic, Axle 1L with Pin 2L with Friction Ridges'
    parts_dict['11212'] = 'Plate 3 x 3'
    parts_dict['11211'] = 'Brick, Modified 1 x 2 with Studs on 1 Side'
    parts_dict['11090'] = 'Bar Holder with Clip'
    parts_dict['10247'] = 'Plate, Modified 2 x 2 with Pin Hole - Full Cross Support Underneath'
    return parts_dict

# After creating a script to scrape images, several parts were not found. These not-found parts appear to have been
# referenced in the B200C by a less common part number that has a more-common equivalent.
#
# Below is a list of the not-found parts and the more-common equivalent part numbers.
# Where the allowable_parts is an OrderedDict, for sampling purposes, this is just used for item lookup, so
# a standard dict can be used here.
def alternative_part_numbers() -> dict:
    alternatives = dict()
    alternatives['3062b'] = '3062'
    alternatives['3068b'] = '3068'
    alternatives['3069b'] = '3069'
    alternatives['3070b'] = '3070'
    alternatives['15379'] = '3873'
    alternatives['32123'] = '4265c'
    alternatives['59443'] = '6538c'
    alternatives['88072'] = '4623b'
    alternatives['88323'] = '57518'
    return alternatives

