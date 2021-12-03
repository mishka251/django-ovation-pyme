from ovation_prime_app.my_types import CoordinatesValue


def round_coordinates_tuple(coords: CoordinatesValue) -> CoordinatesValue:
    lat, lon, val = coords
    lat = round(lat, 0)
    lon = round(lon, 0)
    val = round(val, 1)
    return CoordinatesValue(lat, lon, val)


def round_coordinates(coordinates: 'list[CoordinatesValue]') -> 'list[CoordinatesValue]':
    return list(map(round_coordinates_tuple, coordinates))
