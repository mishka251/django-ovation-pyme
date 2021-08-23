import datetime
import cProfile
# import the logging library
import logging

# Get an instance of a logger
logger = logging.getLogger(__name__)

from django.http import JsonResponse
from django.shortcuts import render
from django.conf import settings
# Create your views here.

from ovationpyme import ovation_prime


def get_mlat_grid():
    return settings.MLAT_GRID


def get_mlt_grid():
    return settings.MLT_GRID


def get_ovation_prime_data(request):
    """
    Отдаем json с данными для построения одного из графиков
    Параметры запроса :hemi: - полушарие "N" или "S"
    :dt: - датавремя в формате `yyyy-mm-ddTHH:MM:SS`
    """

    with cProfile.Profile() as pf:

        hemi = request.GET.get('hemi', 'N')
        _dt = request.GET.get('dt', '')

        dt = datetime.datetime.fromisoformat(_dt)

        new_mlat_grid = get_mlat_grid()
        new_mlt_grid = get_mlt_grid()

        estimator = ovation_prime.ConductanceEstimator(fluxtypes=['diff', 'mono'])

        mlatgrid, mltgrid, pedgrid, hallgrid = estimator.get_conductance(dt, hemi=hemi, auroral=True, solar=True)

        ped_interpolator = ovation_prime.LatLocaltimeInterpolator(mlatgrid, mltgrid, pedgrid)
        new_pedgrid = ped_interpolator.interpolate(new_mlat_grid, new_mlt_grid)

        hall_interpolator = ovation_prime.LatLocaltimeInterpolator(mlatgrid, mltgrid, hallgrid)
        new_hallgrid = hall_interpolator.interpolate(new_mlat_grid, new_mlt_grid)

        (n, m) = new_mlt_grid.shape

        pedgrid_data = []
        for i in range(n):
            for j in range(m):
                item = {
                    'mlat': new_mlat_grid[i, j],
                    'mlt': new_mlt_grid[i, j],
                    'value': new_pedgrid[i, j],
                }
                pedgrid_data.append(item)
        hallgrid_data = []
        for i in range(n):
            for j in range(m):
                item = {
                    'mlat': new_mlat_grid[i, j],
                    'mlt': new_mlt_grid[i, j],
                    'value': new_hallgrid[i, j],
                }
                hallgrid_data.append(item)

        data = {
            'pedgrid': pedgrid_data,
            'hallgrid': hallgrid_data,
        }

    now = datetime.datetime.now()
    now_str = now.strftime('%Y_%m_%d_%H_%M_%S_%f')

    pf.dump_stats(f'profile_{now_str}.pstat')

    logger.debug('success calculated')

    data_type = 'hallgrid'
    data = data[data_type]

    def parse(data):
        value = data['value']
        mlat = data['mlat']
        mlt = data['mlt']
        latitude = mlat
        longitude = mlt * 15 - 180
        return [longitude, latitude, value]

    parsed_data = [parse(val) for val in data]

    result = {
        "Observation Time": now_str,
        "Forecast Time": _dt,
        "Data Format": f"[Longitude, Latitude, {data_type}]",
        "coordinates": parsed_data
    }

    return JsonResponse(result, safe=False)
