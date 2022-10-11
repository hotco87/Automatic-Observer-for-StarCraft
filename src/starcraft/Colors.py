from matplotlib import colors

colors_in_starcraft = {
    'Red': '#f40404',
    'Blue': '#0c48cc',
    'Teal': '#2cb494',
    'Purple': '#88409c',
    'Orange': '#f88c14',
    'Brown': '#703014',
    'White': '#cce0c0',
    'Yellow': '#fcfc38',
    'Green': '#088008',
    'Pale Yellow': '#fcfc7c',
    'Tan': '#ecc4b0',
    'Dark Aqua': '#4068d4',
    'Pale Green': '#74a47c',
    'Blueish Grey': '#7290b8',
    'Pale Yellow': '#fcfc7c',
    'Cyan': '#00e4fc',
    'Pink': '#ffc4e4',
    'Olive': '#808000',
    'Lime': '#d2f53c',
    'Navy': '#000080',
    'Margenta': '#f032e6',
    'Grey': '#808080',
    'Black': '#3c3c3c'
}

class custom_colors():
    def __init__(self, color: str):
        self._color = color
        
    @property
    def hexCode(self):
        return colors_in_starcraft[self._color]