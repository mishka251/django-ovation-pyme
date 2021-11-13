# 2021.08.26 22:04
import datetime

# Get an instance of a logger
import math

# from ovation_prime_app.types import OvationPrimeData
from ovation_prime_app.my_types import CoordinatesValue, OvationPrimeData
from ovation_prime_app.utils.mag_to_geo import mag_to_geo

import aacgmv2
# import the logging library

def parse(data: 'OvationPrimeData', dt: datetime) -> 'CoordinatesValue':
    value = data['value']
    mlat = data['mlat']
    mlt = data['mlt']
    # latitude = mlat_to_lat(mlat)
    # longitude = mlt_to_lon(mlt)
    mlon = aacgmv2.convert_mlt(mlt, dt, True)
    if mlon < 0:
        mlon += 360
    # mlon2 = (mlt * 24 + 180) % 360

    # print(mlon, mlon2)

    # test = smToLatLon([mlat], [mlon], dt)
    test2 = mag_to_geo(mlat, mlon, dt)

    # print(test, [longitude, latitude])
    # longitude = test[1][0]
    # latitude = test[0][0]

    longitude = math.degrees(test2[1])
    latitude = math.degrees(test2[0])
    # if latitude < 0:
    #     latitude += 360
    return CoordinatesValue(longitude, latitude, value)
