import re
import typing
from pathlib import Path


VALUE_NAME_MAP = {
    'baseColor': 'base_color',
    'opacity': 'opacity',
    'glow': 'glow',
    'roughness': 'roughness',
    'metallic': 'metallic',
    'translucence': 'translucence',
    'density': 'density',
    'interiorColor': 'subsurface_color',
    'normal': 'normal',
    'height': 'height',
    'heightScale': 'height_scale',
    'indexOfRefraction': 'ior',
}


VALUE_TEXTURE_2D_PATTERN = r'texture_2d\(\"(.+)\"'
VALUE_COLOR_PATTERN = r'color\((.+)\)'
VALUE_FLOAT_PATTERN = r'(\d*\.?\d*?)f?'


def _parse_file_texture(desc):
    if 'texture_2d' in desc:
        pattern = re.compile(VALUE_TEXTURE_2D_PATTERN)
        m = pattern.search(desc)
        return m.group(1)

    raise RuntimeError('Only texture_2d is supported for file_texture')


def _parse_color(desc):
    pattern = re.compile(VALUE_COLOR_PATTERN)
    m = pattern.search(desc)
    if m is None:
        raise RuntimeError('Could not parse color')

    color_str = m.group(1)
    # print(color_str)
    colors = tuple(float(s.replace('f', '')) for s in re.split(r', ?', color_str))

    if len(colors) not in {3, 4}:
        raise RuntimeError(f'Invalid number of color channels: {len(colors)}')

    if len(colors) == 3:
        colors = (*colors, 1.0)

    return colors


def _parse_value(value_str):
    if value_str.startswith('::base::file_texture'):
        return _parse_file_texture(value_str)
    if value_str.startswith('::base::tangent_space_normal_texture'):
        return _parse_file_texture(value_str)
    elif '::state::normal()' in value_str:
        return None
    elif value_str.startswith('color'):
        return _parse_color(value_str)
    elif re.match(VALUE_FLOAT_PATTERN, value_str):
        match = re.match(VALUE_FLOAT_PATTERN, value_str).group(1)
        return float(match)
    else:
        raise RuntimeError(f'Input type not supported: {value_str!r}')


def _find_value(key, mdl):
    pattern = re.compile(rf'{key}: ?(.*),?\)?\n')
    m = pattern.search(mdl)
    if m is None:
        return None
    return  _parse_value(m.group(1))


def parse_mdl(path: typing.Union[str, Path]):
    if isinstance(path, str):
        path = Path(path)

    with open(path, 'r') as f:
        mdl = f.read()

    parsed_dict = {}

    for value_key, value_new_key in VALUE_NAME_MAP.items():
        value = _find_value(value_key, mdl)
        if value is not None:
            parsed_dict[value_new_key] = value
    return parsed_dict
