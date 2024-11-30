from typing import Dict, Tuple
import colorsys

# Color codes adapted from the list provided by Rebrickable:
# https://rebrickable.com/colors/
# This list features only colors available on or after 1988, which is the oldest realistic date for bricks that might
# exist in the collection from which the samples were drawn. Additionally, only colors that were made for at least 40
# parts were considered (i.e., no colors specific to specialty sets)
def get_colors() -> Dict[str, str]:
    colors = dict()
    colors['Light Salmon'] = 'FEBABD'
    colors['Light Pink'] = 'FECCCF'
    colors['Duplo Pink'] = 'FF879C'
    colors['Pink'] = 'FC97AC'
    colors['Coral'] = 'FF698F'
    colors['Medium Dark Pink'] = 'F785B1'
    colors['Glitter Trans-Dark Pink'] = 'DF6695'
    colors['Light Purple'] = 'CD6298'
    colors['Bright Pink'] = 'E4ADC8'
    colors['Dark Pink'] = 'C870A0'
    colors['Opal Trans-Dark Pink'] = 'CE1D9B'
    colors['Magenta'] = '923978'
    colors['Purple'] = '81007B'
    colors['Sand Purple'] = '845E84'
    colors['Reddish Lilac'] = '8E5597'
    colors['Trans-Light Purple'] = '96709F'
    colors['Medium Lavender'] = 'AC78BA'
    colors['Opal Trans-Purple'] = '8320B7'
    colors['Lavender'] = 'E1D5ED'
    colors['Trans-Medium Purple'] = '8D73B3'
    colors['Dark Purple'] = '3F3691'
    colors['Glitter Trans-Purple'] = 'A5A5CB'
    colors['Light Violet'] = 'C9CAE2'
    colors['Light Lilac'] = '9195CA'
    colors['Medium Bluish Violet'] = '6874CA'
    colors['Royal Blue'] = '4C61DB'
    colors['Opal Trans-Dark Blue'] = '0020A0'
    colors['Violet'] = '4354A3'
    colors['Sand Blue'] = '6074A1'
    colors['Pearl Sand Blue'] = '7988A1'
    colors['Blue'] = '0055BF'
    colors['Medium Blue'] = '5A93DB'
    colors['Dark Blue'] = '0A3463'
    colors['Trans-Light Royal Blue'] = 'B4D4F7'
    colors['Trans-Medium Blue'] = 'CFE2F7'
    colors['Bright Light Blue'] = '9FC3E9'
    colors['Light Blue'] = 'B4D2E3'
    colors['Trans-Very Lt Blue'] = 'C1DFF0'
    colors['Maersk Blue'] = '3592C3'
    colors['Dark Azure'] = '078BC9'
    colors['Sky Blue'] = '7DBFDD'
    colors['Pastel Blue'] = '5AC4DA'
    colors['Medium Azure'] = '36AEBF'
    colors['Light Turquoise'] = '55A5AF'
    colors['Glitter Trans-Light Blue'] = '68BCC5'
    colors['Dark Turquoise'] = '008F9B'
    colors['Trans-Light Blue'] = 'AEEFEC'
    colors['Light Aqua'] = 'ADC3C0'
    colors['Aqua'] = 'B3D7D1'
    colors['Dark Green'] = '184632'
    colors['Medium Green'] = '73DCA1'
    colors['Sand Green'] = 'A0BCAC'
    colors['Green'] = '237841'
    colors['Trans-Green'] = '84B68D'
    colors['Bright Green'] = '4B9F4A'
    colors['Light Green'] = 'C2DAB8'
    colors['Glow In Dark Trans'] = 'BDC6AD'
    colors['Trans-Light Bright Green'] = 'C9E788'
    colors['Pearl Lime'] = '6A7944'
    colors['Lime'] = 'BBE90B'
    colors['Yellowish Green'] = 'DFEEA5'
    colors['Trans-Bright Green'] = 'D9E4A7'
    colors['Medium Lime'] = 'C7D23C'
    colors['Olive Green'] = '9B9A5A'
    colors['Trans-Neon Green'] = 'F8F184'
    colors['Vibrant Yellow'] = 'EBD800'
    colors['Bright Light Yellow'] = 'FFF03A'
    colors['Chrome Gold'] = 'BBA53D'
    colors['Trans-Fire Yellow'] = 'FBE890'
    colors['Trans-Neon Yellow'] = 'DAB000'
    colors['Trans-Yellow'] = 'F5CD2F'
    colors['Yellow'] = 'F2CD37'
    colors['Light Yellow'] = 'FBE696'
    colors['Metallic Gold'] = 'DBAC34'
    colors['Bright Light Orange'] = 'F8BB3D'
    colors['Tan'] = 'E4CD9E'
    colors['Dark Tan'] = '958A73'
    colors['Pearl Gold'] = 'AA7F2E'
    colors['Medium Orange'] = 'FFA70B'
    colors['Dark Brown'] = '352100'
    colors['Pearl Copper'] = 'B46A00'
    colors['Earth Orange'] = 'FA9C1C'
    colors['Light Orange'] = 'F9BA61'
    colors['Reddish Gold'] = 'AC8247'
    colors['Trans-Orange'] = 'F08F1C'
    colors['Warm Tan'] = 'CCA373'
    colors['Light Nougat'] = 'F6D7B3'
    colors['Trans-Flame Yellowish Orange'] = 'FCB76D'
    colors['Dark Orange'] = 'A95500'
    colors['Orange'] = 'FE8A18'
    colors['Flat Dark Gold'] = 'B48455'
    colors['Trans-Neon Orange'] = 'FF800D'
    colors['Medium Nougat'] = 'AA7D55'
    colors['Fabuland Brown'] = 'B67B50'
    colors['Medium Brown'] = '755945'
    colors['Nougat'] = 'D09168'
    colors['Brown'] = '583927'
    colors['Copper'] = 'AE7A59'
    colors['Reddish Brown'] = '582A12'
    colors['Reddish Orange'] = 'CA4C0B'
    colors['Light Brown'] = '7C503A'
    colors['Metallic Copper'] = '764D3B'
    colors['Salmon'] = 'F2705E'
    colors['Red'] = 'C91A09'
    colors['Sand Red'] = 'D67572'
    colors['Milky White'] = 'FFFFFF'
    colors['Trans-Clear'] = 'FCFCFC'
    colors['Pearl White'] = 'F2F3F2'
    colors['Very Light Gray'] = 'E6E3DA'
    colors['Very Light Bluish Gray'] = 'E6E3E0'
    colors['Chrome Silver'] = 'E0E0E0'
    colors['Glow In Dark Opaque'] = 'D4D5C9'
    colors['Metal'] = 'A5ADB4'
    colors['Metallic Silver'] = 'A5A9B4'
    colors['Light Bluish Gray'] = 'A0A5A9'
    colors['Pearl Light Gray'] = '9CA3A8'
    colors['Light Gray'] = '9BA19D'
    colors['Flat Silver'] = '898788'
    colors['Dark Gray'] = '6D6E5C'
    colors['Dark Bluish Gray'] = '6C6E68'
    colors['Trans-Brown'] = '635F52'
    colors['Pearl Dark Gray'] = '575857'
    colors['Pearl Titanium'] = '3E3C39'
    colors['Black'] = '05131D'
    colors['Glow in Dark White'] = 'D9D9D9'
    return colors


def get_colors_hsv() -> Dict[str, Tuple[float, float, float]]:
    colors = get_colors()
    colors_hsv = dict()
    for color in colors.items():
        colors_hsv[color[0]] = hex_to_hsv(color[1])
    return colors_hsv


# This function based of Google's AI Overview result for the query "rgb hex to hsv conversion python", searched
# on 29NOV2024.
def hex_to_hsv(hex_str: str) -> Tuple[int, int, int]:
    """
    Converts a hex color code to HSV values.

    @param hex_str: RGB color code in hex string format
    @return: HSV color value as a tuple(int, int, int), ranging 0-255
    """

    # Remove the '#' if it exists
    hex_color = hex_str.lstrip('#')

    # Convert hex to RGB
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0

    # Convert RGB to HSV
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    return int(h*255.0), int(s*255.0), int(v*255.0)