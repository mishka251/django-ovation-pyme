# 2021.08.26 22:04
import datetime
import cProfile
# import the logging library
import logging

# Get an instance of a logger
import math
from functools import cmp_to_key
from typing import TypedDict, List

import numpy as np
from django.forms import Form, DateTimeField, ChoiceField, DecimalField
from django.forms.utils import ErrorDict

logger = logging.getLogger(__name__)

from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.conf import settings
# Create your views here.

import aacgmv2

from ovationpyme import ovation_prime
from auromat.coordinates.transform import smToLatLon


class OvationPrimeData(TypedDict):
    value: float
    mlt: float
    mlat: float


def get_north_mlat_grid():
    return settings.NORTH_MLAT_GRID


def get_north_mlt_grid():
    return settings.NORTH_MLT_GRID


def get_south_mlat_grid():
    return settings.SOUTH_MLAT_GRID


def get_south_mlt_grid():
    return settings.SOUTH_MLT_GRID


def mlt_to_lon(mlt: float) -> float:
    return mlt * 15 - 180


def mlat_to_lat(mlat: float) -> float:
    return mlat


def mag_to_geo(latMAG: float, longMAG: float) -> tuple[float, float, float]:
    mag1_sgn = 1
    if longMAG > 180:
        longMAG = 360 - longMAG
        mag1_sgn = -1

    MAG = [0, 0, 0]

    cos = math.cos(math.radians(longMAG))
    tan = math.tan(math.radians(latMAG))

    cos2 = cos * cos
    tan2 = tan * tan

    MAG[0] = cos2 * (1 - tan2 / (1 + tan2))
    MAG[1] = MAG[0] * (1 / cos2 - 1)
    MAG[2] = tan2 / (1 + tan2)

    MAG[0] = math.sqrt(MAG[0])
    MAG[1] = math.sqrt(MAG[1])
    MAG[2] = math.sqrt(MAG[2])
    if mag1_sgn == -1:
        MAG[1] *= -1
    if tan < 0:
        MAG[2] *= -1
    if cos < 0:
        MAG[0] *= -1

    h11 = 4797.1  #
    g11 = -1501  # Сферические гармонические коэффициенты для эпохи 2015-2020
    g10 = -29442  #
    lamda = math.atan(h11 / g11)  # угол поворотота вокруг оси Y
    fi = - math.asin((g11 * math.cos(lamda) + h11 * math.sin(lamda)) / (g10))

    t5Y = [
        [math.cos(fi), 0, math.sin(fi)],
        [0, 1, 0],
        [-math.sin(fi), 0, math.cos(fi)]
    ]

    # print(t5Y)

    t5Z = [
        [math.cos(lamda), math.sin(lamda), 0],
        [-math.sin(lamda), math.cos(lamda), 0],
        [0, 0, 1]
    ]

    # print(t5Z)

    t5 = np.dot(t5Y, t5Z)
    t5_inv = np.linalg.inv(t5)
    GEO = np.dot(t5_inv, MAG)

    rE = math.sqrt(GEO[0] * GEO[0] + GEO[1] * GEO[1] + GEO[2] * GEO[2])

    x = GEO[0] / rE  # math.cos(latGEO) * math.cos(longGEO)
    y = GEO[1] / rE  # math.cos(latGEO) * math.sin(longGEO)
    z = GEO[2] / rE  # math.sin(latGEO)

    _latGEO1 = math.asin(z)
    if _latGEO1 > 0:
        _latGEO2 = math.pi - _latGEO1
    else:
        _latGEO2 = -math.pi - _latGEO1

    cos_lat1 = math.cos(_latGEO1)
    cos_lat2 = math.cos(_latGEO2)

    long_sin1 = y / cos_lat1
    if long_sin1 > 1:
        long_sin1 = 1
    if long_sin1 < -1:
        long_sin1 = -1
    long_sin2 = y / cos_lat2
    if long_sin2 > 1:
        long_sin2 = 1
    if long_sin2 < -1:
        long_sin2 = -1

    _longGEO1 = math.asin(long_sin1)
    _longGEO3 = math.asin(long_sin2)

    _x1 = math.cos(_latGEO1) * math.cos(_longGEO1)
    _x3 = math.cos(_latGEO2) * math.cos(_longGEO3)
    eps = 1e-6
    if abs(x - _x1) < eps:
        return _latGEO1, _longGEO1, 0
    elif abs(x - _x3) < eps:
        return _latGEO2, _longGEO3, 0
    else:
        raise ValueError()


def parse(data: OvationPrimeData, dt: datetime) -> [float, float, float]:
    value = data['value']
    mlat = data['mlat']
    mlt = data['mlt']
    latitude = mlat_to_lat(mlat)
    longitude = mlt_to_lon(mlt)
    mlon = aacgmv2.convert_mlt(mlt, dt, True)
    if mlon < 0:
        mlon += 360
    mlon2 = (mlt * 24 + 180) % 360

    print(mlon, mlon2)

    test = smToLatLon([mlat], [mlon], dt)
    test2 = mag_to_geo(mlat, mlon)

    # print(test, [longitude, latitude])
    # longitude = test[1][0]
    # latitude = test[0][0]

    longitude = math.degrees(test2[1])
    latitude = math.degrees(test2[0])
    if latitude < 0:
        latitude += 360
    return [longitude, latitude, value]


def sort_coordinates(coords: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    def comparator(a: [float, float, float], b: [float, float, float]) -> int:
        for i in range(2):
            if a[i] > b[i]:
                return 1
            if a[i] < b[i]:
                return -1
        return 0

    return list(sorted(coords, key=cmp_to_key(comparator)))


def grids_to_dicts(mlat_grid, mlt_grid, value_grid) -> List[OvationPrimeData]:
    (n, m) = mlat_grid.shape

    result = []
    for i in range(n):
        for j in range(m):
            item = {
                'mlat': mlat_grid[i, j],
                'mlt': mlt_grid[i, j],
                'value': value_grid[i, j],
            }
            result.append(item)
    return result


class OvationPrimeConductanceForm(Form):
    dt = DateTimeField(required=True)
    _type = ChoiceField(choices=[('pedgrid', 'pedgrid'), ('hallgrid', 'hallgrid')], required=True)


def plot(new_mlat_grid, new_mlt_grid, vals, hemi, dt, view_name: str):
    import matplotlib.pyplot as pp
    from geospacepy import satplottools

    f = pp.figure(figsize=(11, 5))
    aH = f.add_subplot(111)
    # aP = f.add_subplot(122)

    X, Y = satplottools.latlt2cart(new_mlat_grid.flatten(), new_mlt_grid.flatten(), hemi)
    X = X.reshape(new_mlat_grid.shape)
    Y = Y.reshape(new_mlt_grid.shape)

    satplottools.draw_dialplot(aH)
    # satplottools.draw_dialplot(aP)

    # mappableH = aH.pcolormesh(X, Y, new_hallgrid, vmin=0., vmax=20.)
    # mappableP = aP.pcolormesh(X, Y, new_pedgrid, vmin=0., vmax=15.)

    mappableH = aH.pcolormesh(X, Y, vals, vmin=0., vmax=20.)

    aH.set_title("Hall Conductance")
    # aP.set_title("Pedersen Conductance")

    f.colorbar(mappableH, ax=aH)
    # f.colorbar(mappableP, ax=aP)

    f.suptitlef("{2} {0} Hemisphere at {1}".format(hemi, dt.strftime('%c'), view_name),
                fontweight='bold')
    f.savefig('{2}_{1}_{0}.png'.format(dt.strftime('%Y%m%d_%H%M%S'), hemi, view_name))

    # return f


def get_ovation_prime_conductance_interpolated(request):
    """
    Отдаем json с данными для построения одного из графиков
    Параметры запроса
    :dt: - датавремя в формате `yyyy-mm-ddTHH:MM:SS`
    :type: тип расчётов? 'pedgrid' или  'hallgrid'
    """
    form = OvationPrimeConductanceForm(request.GET)
    is_valid = form.is_valid()
    if not is_valid:
        return HttpResponseBadRequest(form.errors.as_json())

    dt = form.cleaned_data['dt']
    _type = form.cleaned_data['_type']

    new_north_mlat_grid = get_north_mlat_grid()
    new_north_mlt_grid = get_north_mlt_grid()

    new_south_mlat_grid = get_south_mlat_grid()
    new_south_mlt_grid = get_south_mlt_grid()

    estimator = ovation_prime.ConductanceEstimator(fluxtypes=['diff', 'mono'])

    north_mlatgrid, north_mltgrid, north_pedgrid, north_hallgrid, oi = estimator.get_conductance(dt, hemi='N',
                                                                                                 auroral=True,
                                                                                                 solar=True)

    south_mlatgrid, south_mltgrid, south_pedgrid, south_hallgrid, oi = estimator.get_conductance(dt, hemi='S',
                                                                                                 auroral=True,
                                                                                                 solar=True)

    if _type == 'pedgrid':
        north_interpolator = ovation_prime.LatLocaltimeInterpolator(north_mlatgrid, north_mltgrid, north_pedgrid)
        north_new_values = north_interpolator.interpolate(new_north_mlat_grid, new_north_mlt_grid)

        south_interpolator = ovation_prime.LatLocaltimeInterpolator(south_mlatgrid, south_mltgrid, south_pedgrid)
        south_new_values = south_interpolator.interpolate(new_south_mlat_grid, new_south_mlt_grid)

    else:
        north_interpolator = ovation_prime.LatLocaltimeInterpolator(north_mlatgrid, north_mltgrid, north_hallgrid)
        north_new_values = north_interpolator.interpolate(new_north_mlat_grid, new_north_mlt_grid)

        south_interpolator = ovation_prime.LatLocaltimeInterpolator(south_mlatgrid, south_mltgrid, south_hallgrid)
        south_new_values = south_interpolator.interpolate(new_south_mlat_grid, new_south_mlt_grid)

    plot(new_north_mlat_grid, new_north_mlt_grid, north_new_values, 'N', dt, "conductance_interpolated")
    plot(new_south_mlat_grid, new_south_mlt_grid, south_new_values, 'S', dt, "conductance_interpolated")

    _data = [
        *grids_to_dicts(new_north_mlat_grid, new_north_mlt_grid, north_new_values),
        *grids_to_dicts(new_south_mlat_grid, new_south_mlt_grid, south_new_values),
    ]

    now = datetime.datetime.now()
    now_str = now.strftime('%Y_%m_%d_%H_%M_%S_%f')

    parsed_data = [parse(val, dt) for val in _data]
    # parsed_data = sort_coordinates(parsed_data)

    print(oi)
    # parsed_data = [
    #     [55, 56, 10],
    #     [56, 56, 20],
    #     [55, 55, 2],
    #     [55, -55, 15],
    #     [-55, 55, 0.5],
    #     [-55, -55, -2],
    # ]

    result = {
        "Observation Time": [str(oi.startdt), str(oi.enddt)],
        "Forecast Time": str(dt),
        "Data Format": f"[Longitude, Latitude, {_type}]",
        "coordinates": parsed_data
    }

    logger.debug('success calculated')
    return JsonResponse(result, safe=False)


def get_ovation_prime_conductance(request):
    """
    Отдаем json с данными для построения одного из графиков
    Параметры запроса
    :dt: - датавремя в формате `yyyy-mm-ddTHH:MM:SS`
    :type: тип расчётов? 'pedgrid' или  'hallgrid'
    """

    form = OvationPrimeConductanceForm(request.GET)
    is_valid = form.is_valid()
    if not is_valid:
        return HttpResponseBadRequest(form.errors.as_json())

    dt = form.cleaned_data['dt']
    _type = form.cleaned_data['_type']

    estimator = ovation_prime.ConductanceEstimator(fluxtypes=['diff', 'mono'])

    north_mlatgrid, north_mltgrid, north_pedgrid, north_hallgrid, oi = estimator.get_conductance(dt, hemi='N',
                                                                                                 auroral=True,
                                                                                                 solar=True)

    south_mlatgrid, south_mltgrid, south_pedgrid, south_hallgrid, oi = estimator.get_conductance(dt, hemi='S',
                                                                                                 auroral=True,
                                                                                                 solar=True)

    if _type == 'pedgrid':
        north_data = north_pedgrid
        south_data = south_pedgrid
    else:
        north_data = north_hallgrid
        south_data = south_hallgrid

    plot(north_mlatgrid, north_mltgrid, north_data, 'N', dt, "conductance")
    plot(south_mlatgrid, south_mltgrid, south_data, 'S', dt, "conductance")

    _data = [
        *grids_to_dicts(north_mlatgrid, north_mltgrid, north_data),
        *grids_to_dicts(south_mlatgrid, south_mltgrid, south_data),
    ]

    now = datetime.datetime.now()
    now_str = now.strftime('%Y_%m_%d_%H_%M_%S_%f')

    parsed_data = [parse(val, dt) for val in _data]

    print(oi)

    result = {
        "Observation Time": [str(oi.startdt), str(oi.enddt)],
        "Forecast Time": str(dt),
        "Data Format": f"[Longitude, Latitude, {_type}]",
        "coordinates": parsed_data
    }

    logger.debug('success calculated')
    return JsonResponse(result, safe=False)


class WeightedFluxForm(Form):
    dt = DateTimeField(required=True)
    atype = ChoiceField(choices=[('diff', 'diff'), ('mono', 'mono'), ('wave', 'wave'), ('ions', 'ions')],
                        initial='diff', required=False)
    jtype = ChoiceField(choices=[('energy', 'energy'), ('number', 'number')], initial='energy', required=False)


def get_weighted_flux(request):
    """
    Отдаем json с данными для построения графиков из
    draw_weighted_flux
    Параметры запроса
    :dt: - датавремя в формате `yyyy-mm-ddTHH:MM:SS`
    :atype: - str, ['diff','mono','wave','ions']
            type of aurora for which to load regression coeffients
    :jtype: - str, ['energy','number']
            Type of flux you want to estimate
    """
    form = WeightedFluxForm(request.GET)
    is_valid = form.is_valid()
    if not is_valid:
        return HttpResponseBadRequest(form.errors.as_json())

    dt = form.cleaned_data['dt']
    atype = form.cleaned_data['atype'] or form.fields['atype'].initial
    jtype = form.cleaned_data['jtype'] or form.fields['jtype'].initial

    estimator = ovation_prime.FluxEstimator(atype, jtype)

    mlatgridN, mltgridN, fluxgridN, oi = estimator.get_flux_for_time(dt, hemi='N')
    mlatgridS, mltgridS, fluxgridS, oi = estimator.get_flux_for_time(dt, hemi='S')

    _data = [
        *grids_to_dicts(mlatgridN, mltgridN, fluxgridN),
        *grids_to_dicts(mlatgridS, mltgridS, fluxgridS),
    ]

    now = datetime.datetime.now()
    now_str = now.strftime('%Y_%m_%d_%H_%M_%S_%f')

    parsed_data = [parse(val, dt) for val in _data]

    result = {
        "Observation Time": [str(oi.startdt), str(oi.enddt)],
        "Forecast Time": str(dt),
        "Data Format": f"[Longitude, Latitude, weighted_flux]",
        "coordinates": parsed_data
    }

    logger.debug('success calculated')
    return JsonResponse(result, safe=False)


season_choices = [
    ('winter', 'winter'),
    ('spring', 'spring'),
    ('summer', 'summer'),
    ('fall', 'fall'),
]


class SeasonalFluxForm(Form):
    dt = DateTimeField(required=True)
    atype = ChoiceField(choices=[('diff', 'diff'), ('mono', 'mono'), ('wave', 'wave'), ('ions', 'ions')],
                        initial='diff', required=False)
    jtype = ChoiceField(choices=[('energy', 'energy'), ('number', 'number')], initial='energy', required=False)

    seasonN = ChoiceField(choices=season_choices, initial='summer', required=False)
    seasonS = ChoiceField(choices=season_choices, initial='winter', required=False)

    dF = DecimalField(required=False, initial=2134.17)


def get_seasonal_flux(request):
    """
    Отдаем json с данными для построения графиков из
    draw_weighted_flux
    Параметры запроса
    :dt: - датавремя в формате `yyyy-mm-ddTHH:MM:SS`
    :atype: - str, ['diff','mono','wave','ions']
            type of aurora for which to load regression coeffients
    :jtype: - str, ['energy','number']
            Type of flux you want to estimate
    :seasonN: str ['winter', 'summer', 'spring', 'fall']
    :seasonS: str ['winter', 'summer', 'spring', 'fall']
    """
    form = SeasonalFluxForm(request.GET)
    is_valid = form.is_valid()
    if not is_valid:
        return HttpResponseBadRequest(form.errors.as_json())

    dt = form.cleaned_data['dt']
    atype = form.cleaned_data['atype'] or form.fields['atype'].initial
    jtype = form.cleaned_data['jtype'] or form.fields['jtype'].initial

    season_n = form.cleaned_data['seasonN'] or form.fields['seasonN'].initial
    season_s = form.cleaned_data['seasonS'] or form.fields['seasonS'].initial

    dF = form.cleaned_data['dF'] or form.fields['dF'].initial

    estimatorN = ovation_prime.SeasonalFluxEstimator(season_n, atype, jtype)
    estimatorS = ovation_prime.SeasonalFluxEstimator(season_s, atype, jtype)

    fluxtupleN = estimatorN.get_gridded_flux(dF, combined_N_and_S=False)
    (mlatgridN, mltgridN, fluxgridN) = fluxtupleN[:3]

    fluxtupleS = estimatorS.get_gridded_flux(dF, combined_N_and_S=False)
    (mlatgridS, mltgridS, fluxgridS) = fluxtupleS[3:]

    _data = [
        *grids_to_dicts(mlatgridN, mltgridN, fluxgridN),
        *grids_to_dicts(mlatgridS, mltgridS, fluxgridS),
    ]

    now = datetime.datetime.now()
    now_str = now.strftime('%Y_%m_%d_%H_%M_%S_%f')

    parsed_data = [parse(val, dt) for val in _data]

    result = {
        "Observation Time": now_str,
        "Forecast Time": str(dt),
        "Data Format": f"[Longitude, Latitude, seasonal_flux]",
        "coordinates": parsed_data
    }

    logger.debug('success calculated')
    return JsonResponse(result, safe=False)
