# 2021.08.26 22:04
import datetime
import cProfile
# import the logging library
import logging

# Get an instance of a logger
from typing import TypedDict, List

from django.forms import Form, DateTimeField, ChoiceField, DecimalField
from django.forms.utils import ErrorDict

logger = logging.getLogger(__name__)

from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.conf import settings
# Create your views here.

from ovationpyme import ovation_prime


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


def parse(data: OvationPrimeData) -> [float, float, float]:
    value = data['value']
    mlat = data['mlat']
    mlt = data['mlt']
    latitude = mlat_to_lat(mlat)
    longitude = mlt_to_lon(mlt)
    return [longitude, latitude, value]


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
        return HttpResponseBadRequest(form.errors.get_json_data())

    dt = form.cleaned_data['dt']
    _type = form.cleaned_data['_type']

    new_north_mlat_grid = get_north_mlat_grid()
    new_north_mlt_grid = get_north_mlt_grid()

    new_south_mlat_grid = get_south_mlat_grid()
    new_south_mlt_grid = get_south_mlt_grid()

    estimator = ovation_prime.ConductanceEstimator(fluxtypes=['diff', 'mono'])

    north_mlatgrid, north_mltgrid, north_pedgrid, north_hallgrid = estimator.get_conductance(dt, hemi='N',
                                                                                             auroral=True,
                                                                                             solar=True)

    south_mlatgrid, south_mltgrid, south_pedgrid, south_hallgrid = estimator.get_conductance(dt, hemi='S',
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

    _data = [
        *grids_to_dicts(new_north_mlat_grid, new_north_mlt_grid, north_new_values),
        *grids_to_dicts(new_south_mlat_grid, new_south_mlt_grid, south_new_values),
    ]

    now = datetime.datetime.now()
    now_str = now.strftime('%Y_%m_%d_%H_%M_%S_%f')

    parsed_data = [parse(val) for val in _data]

    result = {
        "Observation Time": now_str,
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
        return HttpResponseBadRequest(form.errors.get_json_data())

    dt = form.cleaned_data['dt']
    _type = form.cleaned_data['_type']

    estimator = ovation_prime.ConductanceEstimator(fluxtypes=['diff', 'mono'])

    north_mlatgrid, north_mltgrid, north_pedgrid, north_hallgrid = estimator.get_conductance(dt, hemi='N',
                                                                                             auroral=True,
                                                                                             solar=True)

    south_mlatgrid, south_mltgrid, south_pedgrid, south_hallgrid = estimator.get_conductance(dt, hemi='S',
                                                                                             auroral=True,
                                                                                             solar=True)

    if _type == 'pedgrid':
        north_data = north_pedgrid
        south_data = south_pedgrid
    else:
        north_data = north_hallgrid
        south_data = south_mlatgrid

    _data = [
        *grids_to_dicts(north_mlatgrid, north_mltgrid, north_data),
        *grids_to_dicts(south_mlatgrid, south_mltgrid, south_data),
    ]

    now = datetime.datetime.now()
    now_str = now.strftime('%Y_%m_%d_%H_%M_%S_%f')

    parsed_data = [parse(val) for val in _data]

    result = {
        "Observation Time": now_str,
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

    mlatgridN, mltgridN, fluxgridN = estimator.get_flux_for_time(dt, hemi='N')
    mlatgridS, mltgridS, fluxgridS = estimator.get_flux_for_time(dt, hemi='S')

    _data = [
        *grids_to_dicts(mlatgridN, mltgridN, fluxgridN),
        *grids_to_dicts(mlatgridS, mltgridS, fluxgridS),
    ]

    now = datetime.datetime.now()
    now_str = now.strftime('%Y_%m_%d_%H_%M_%S_%f')

    parsed_data = [parse(val) for val in _data]

    result = {
        "Observation Time": now_str,
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

    parsed_data = [parse(val) for val in _data]

    result = {
        "Observation Time": now_str,
        "Forecast Time": str(dt),
        "Data Format": f"[Longitude, Latitude, seasonal_flux]",
        "coordinates": parsed_data
    }

    logger.debug('success calculated')
    return JsonResponse(result, safe=False)

