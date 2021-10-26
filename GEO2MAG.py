import math
from pylab import *  # Подключение пакетов "matplotlib" (matlab-подобный набор команд для построения графиков) и "numpy"
import numpy as np

from igrfcoord import convert


# GEO->MAG (2015-2020)

# latGEO = 1 * (math.pi / 180)  #
# longGEO = 1 * (math.pi / 180)  # координаты в географической системе (GEO), переведенные в радианы
# altGEO = 0.18


def geo_2_mag__initial(latGEO: float, longGEO: float, altGEO: float) -> [float, float]:
    h11 = 4797.1  #
    g11 = -1501  # Сферические гармонические коэффициенты для эпохи 2015-2020
    g10 = -29442  #

    a = 6378.245  # Большая и малая полуось земного ээлипсоида
    b = 6356.863019  # по Красовскому

    #

    rE = math.sqrt(math.pow(altGEO, 2) + 2 * altGEO * math.sqrt(
        math.pow(a, 2) * math.pow(math.cos(latGEO), 2) + math.pow(b, 2) * math.pow(math.sin(latGEO), 2)) + (
                           math.pow(a, 4) * math.pow(math.cos(latGEO), 2) + math.pow(b, 4) * math.pow(
                       math.sin(latGEO), 2)) / (
                           math.pow(a, 2) * math.pow(math.cos(latGEO), 2) + math.pow(b, 2) * math.pow(
                       math.sin(latGEO), 2)))
    # print("***")
    # print(rE)

    # расчет углов поворота
    lamda = math.atan(h11 / g11)  # угол поворотота вокруг оси Y
    fi = ((math.pi / 2) - math.asin((g11 * math.cos(lamda) + h11 * math.sin(lamda)) / (g10))) - math.pi / 2
    # print("***")
    # print(lamda)
    # print(fi)

    # перевод в геоцентрическую систему

    xGEO = rE * math.cos(latGEO) * math.cos(longGEO)
    yGEO = rE * math.cos(latGEO) * math.sin(longGEO)
    zGEO = rE * math.sin(latGEO)

    # матрица координат GEO
    GEO = list(range(3))
    GEO[0] = xGEO
    GEO[1] = yGEO
    GEO[2] = zGEO
    # print("***")
    # print(GEO)

    # Поворотные матрицы 3x3
    t5Y = list(range(3))
    for i in range(3):
        t5Y[i] = list(range(3))

    t5Y[0][0] = math.cos(fi)
    t5Y[0][1] = 0
    t5Y[0][2] = math.sin(fi)
    t5Y[1][0] = 0
    t5Y[1][1] = 1
    t5Y[1][2] = 0
    t5Y[2][0] = -math.sin(fi)
    t5Y[2][1] = 0
    t5Y[2][2] = math.cos(fi)
    # print("***")
    # print(t5Y)

    t5Z = list(range(3))
    for i in range(3):
        t5Z[i] = list(range(3))

    t5Z[0][0] = math.cos(lamda)
    t5Z[0][1] = math.sin(lamda)
    t5Z[0][2] = 0
    t5Z[1][0] = -math.sin(lamda)
    t5Z[1][1] = math.cos(lamda)
    t5Z[1][2] = 0
    t5Z[2][0] = 0
    t5Z[2][1] = 0
    t5Z[2][2] = 1
    # print("***")
    # print(t5Z)

    # Умножение матриц: t5 = t5Y*t5Z
    t5 = list(range(3))
    for i in range(3):
        t5[i] = list(range(3))
    t5[0][0] = (t5Y[0][0] * t5Z[0][0]) + (t5Y[0][1] * t5Z[1][0]) + (t5Y[0][2] * t5Z[2][0])
    t5[0][1] = (t5Y[0][0] * t5Z[0][1]) + (t5Y[0][1] * t5Z[1][1]) + (t5Y[0][2] * t5Z[2][1])
    t5[0][2] = (t5Y[0][0] * t5Z[0][2]) + (t5Y[0][1] * t5Z[1][2]) + (t5Y[0][2] * t5Z[2][2])

    t5[1][0] = (t5Y[1][0] * t5Z[0][0]) + (t5Y[1][1] * t5Z[1][0]) + (t5Y[1][2] * t5Z[2][0])
    t5[1][1] = (t5Y[1][0] * t5Z[0][1]) + (t5Y[1][1] * t5Z[1][1]) + (t5Y[1][2] * t5Z[2][1])
    t5[1][2] = (t5Y[1][0] * t5Z[0][2]) + (t5Y[1][1] * t5Z[1][2]) + (t5Y[1][2] * t5Z[2][2])

    t5[2][0] = (t5Y[2][0] * t5Z[0][0]) + (t5Y[2][1] * t5Z[1][0]) + (t5Y[2][2] * t5Z[2][0])
    t5[2][1] = (t5Y[2][0] * t5Z[0][1]) + (t5Y[2][1] * t5Z[1][1]) + (t5Y[2][2] * t5Z[2][1])
    t5[2][2] = (t5Y[2][0] * t5Z[0][2]) + (t5Y[2][1] * t5Z[1][2]) + (t5Y[2][2] * t5Z[2][2])
    # print("***")
    # print(t5)

    # Умножение матриц: MAG = t5*tGEO
    MAG = list(range(3))
    MAG[0] = (t5[0][0] * GEO[0]) + (t5[0][1] * GEO[1]) + (t5[0][2] * GEO[2])
    MAG[1] = (t5[1][0] * GEO[0]) + (t5[1][1] * GEO[1]) + (t5[1][2] * GEO[2])
    MAG[2] = (t5[2][0] * GEO[0]) + (t5[2][1] * GEO[1]) + (t5[2][2] * GEO[2])
    # print("***")
    # print(MAG)

    # пересчет в  град
    latMAG = math.atan((MAG[2]) / (math.sqrt(pow(MAG[0], 2) + pow(MAG[1], 2)))) * (180 / math.pi)
    if MAG[1] > 0:
        longMAG = math.acos((MAG[0]) / (math.sqrt(pow(MAG[0], 2) + pow(MAG[1], 2)))) * (180 / math.pi)
    else:
        longMAG = 360 - (math.acos((MAG[0]) / (math.sqrt(pow(MAG[0], 2) + pow(MAG[1], 2)))) * (180 / math.pi))
    # print("***")
    # print(latMAG)
    # print(longMAG)

    return latMAG, longMAG


def geo_2_mag_fixed(latGEO: float, longGEO: float, altGEO: float) -> [float, float]:
    h11 = 4797.1  #
    g11 = -1501  # Сферические гармонические коэффициенты для эпохи 2015-2020
    g10 = -29442  #

    rE = 1

    # расчет углов поворота
    lamda = math.atan(h11 / g11)  # угол поворотота вокруг оси Y
    fi = - math.asin((g11 * math.cos(lamda) + h11 * math.sin(lamda)) / (g10))
    # print("***")
    # print(lamda)
    # print(fi)

    # перевод в геоцентрическую систему

    xGEO = rE * math.cos(latGEO) * math.cos(longGEO)
    yGEO = rE * math.cos(latGEO) * math.sin(longGEO)
    zGEO = rE * math.sin(latGEO)

    # print(f'{rE=}')
    # print(f'{math.sin(latGEO)=}')

    GEO = [xGEO, yGEO, zGEO]

    geo_len = GEO[0] * GEO[0] + GEO[1] * GEO[1] + GEO[2] * GEO[2]

    assert abs(geo_len - rE * rE) < 1e-6, f'GEO!=rE, {GEO=}, {geo_len=}, {rE*rE=}'

    t5Y = [
        [math.cos(fi), 0, math.sin(fi)],
        [0, 1, 0],
        [-math.sin(fi), 0, math.cos(fi)]
    ]

    t5Z = [
        [math.cos(lamda), math.sin(lamda), 0],
        [-math.sin(lamda), math.cos(lamda), 0],
        [0, 0, 1]
    ]

    t5 = np.dot(t5Y, t5Z)

    MAG = np.dot(t5, GEO)

    # MAG[0] /= 2
    # MAG[1] /= 2
    # MAG[2] /= 2
    mag_len = sqrt(MAG[0] * MAG[0] + MAG[1] * MAG[1] + MAG[2] * MAG[2])
    assert abs(mag_len - 1) < 1e-6
    # print(f'{mag_len=}')

    # пересчет в  град
    tan = (MAG[2]) / (math.sqrt(pow(MAG[0], 2) + pow(MAG[1], 2)))
    cos = (MAG[0]) / (math.sqrt(pow(MAG[0], 2) + pow(MAG[1], 2)))
    # print(f'{cos=}, {tan=}')
    # print(f'geo_2_mag {MAG=}')

    cos = np.longdouble(cos)
    tan = np.longdouble(tan)

    cos2 = cos * cos
    tan2 = tan * tan
    _MAG = [0, 0, 0]

    _MAG[0] = cos2 * (1 - tan2 / (1 + tan2))
    _MAG[1] = _MAG[0] * (1 / cos2 - 1)
    _MAG[2] = tan2 / (1 + tan2)

    _MAG[0] = sqrt(_MAG[0])
    _MAG[1] = sqrt(_MAG[1])
    _MAG[2] = sqrt(_MAG[2])

    if MAG[1] < 0:
        _MAG[1] *= -1
    if tan < 0:
        _MAG[2] *= -1
    if cos < 0:
        _MAG[0] *= -1

    for i in range(3):
        assert abs(MAG[i] - _MAG[i]) < 1e-6

    # print(f'geo_2_mag {_MAG=}')

    t5_inv = np.linalg.inv(t5)

    _GEO1 = np.dot(t5_inv, MAG)
    _GEO2 = np.dot(t5_inv, _MAG)

    for i in range(3):
        assert abs(GEO[i] - _GEO2[i]) < 1e-6

    _x = _GEO2[0] / rE  # math.cos(latGEO) * math.cos(longGEO)
    _y = _GEO2[1] / rE  # math.cos(latGEO) * math.sin(longGEO)
    _z = _GEO2[2] / rE  # math.sin(latGEO)

    assert abs(_x - math.cos(latGEO) * math.cos(longGEO)) < 1e-6
    assert abs(_y - math.cos(latGEO) * math.sin(longGEO)) < 1e-6
    assert abs(_z - math.sin(latGEO)) < 1e-6

    _latGEO1 = math.asin(_z)
    if abs(_latGEO1 - math.pi / 2) < math.radians(0.0625):
        _latGEO1 += math.radians(0.125)
    if abs(_latGEO1 + math.pi / 2) < math.radians(0.0625):
        _latGEO1 += math.radians(0.125)

    if _latGEO1 > 0:
        _latGEO2 = math.pi - _latGEO1
    else:
        _latGEO2 = -math.pi - _latGEO1

    cos_lat1 = math.cos(_latGEO1)
    cos_lat2 = math.cos(_latGEO2)

    if abs(cos_lat1) < 1e-8:
        long_sin1 = 1
        long_sin2 = -1
    else:
        long_sin1 = _y / cos_lat1
        if long_sin1 > 1:
            long_sin1 = 1
        if long_sin1 < -1:
            long_sin1 = -1
        long_sin2 = _y / cos_lat2
        if long_sin2 > 1:
            long_sin2 = 1
        if long_sin2 < -1:
            long_sin2 = -1

    _longGEO1 = math.asin(long_sin1)
    if _longGEO1 > 0:
        _longGEO2 = math.pi - _longGEO1
    else:
        _longGEO2 = -math.pi - _longGEO1

    _longGEO3 = math.asin(long_sin2)
    if _longGEO3 > 0:
        _longGEO4 = math.pi - _longGEO3
    else:
        _longGEO4 = -math.pi - _longGEO3

    # assert any([abs(latGEO-_latGEO)<1e-3 for _latGEO in [_latGEO1, _latGEO2]])
    # assert any([abs(longGEO - _longGEO) < 1e-3 for _longGEO in [_longGEO1, _longGEO2, _longGEO3, _longGEO4]])

    _x1 = math.cos(_latGEO1) * math.cos(_longGEO1)

    _x2 = math.cos(_latGEO1) * math.cos(_longGEO2)

    _x3 = math.cos(_latGEO2) * math.cos(_longGEO3)

    _x4 = math.cos(_latGEO2) * math.cos(_longGEO4)

    eps = 1e-6
    # if abs(_x - _x1) < eps and abs(_x - _x4) < eps:
    #     if abs(_latGEO1-latGEO)<1e-6:
    #         print('x1')
    #     if abs(_latGEO2-latGEO)<1e-6:
    #         print('x4')
    #     print(f'{latGEO=}, {longGEO=}, {_MAG=}, {_x1=}, {_latGEO1=}, {_longGEO1=}, {_latGEO2=}, {_longGEO4=}')
    # elif abs(_x - _x2) < eps and abs(_x - _x3) < eps:
    #     if abs(_latGEO1-latGEO)<1e-6:
    #         print('x2')
    #     if abs(_latGEO2-latGEO)<1e-6:
    #         print('x3')
    #     print(f'{latGEO=}, {longGEO=}, {_MAG=}, {_x2=}, {_latGEO1=}, {_longGEO2=}, {_latGEO2=}, {_longGEO3=}')
    # else:
    #     raise ValueError()

    # assert abs(_latGEO - latGEO) < 1e-6
    # assert abs(_longGEO - longGEO) < 1e-6

    # print(_GEO1)

    # 1) MAG0^2 + MAG1^2 + MAG2^2 = rE^2
    # 2) MAG0^2 + MAG1^2 = MAG2^2 / tan^2
    # 3) MAG0^2 + MAG1^2 = MAG0^2 / cos^2

    # MAG0^2 = MAG1^2/(1/cos^2-1) = MAG1^2*(cos^2)/(1-cos^2) (3)
    # MAG1^2*(cos^2)/(1-cos^2) + MAG1^2 + MAG2^2 = rE^2 (1)
    # MAG1^2 / (1-cos^2) + MAG2^2 = re^2 (1)
    # MAG1^2*(cos^2)/(1-cos^2)  + MAG1^2 = MAG2^2 / tan^2 (2)

    # MAG2^2 / tan^2 + MAG2^2 - re^2
    # MAG2^2 * (1+1/tan^2) = re^2
    # MAG2^2 * ((tan^2+1)(tan^2)) = re^2
    # MAG2^2 = (tan^2)/(1+tan^2)
    # MAG0^2/cos2 = rE^2 - MAG2^2
    # MAG0^2 = cos^2 * (1 - MAG2^2)

    # MAG1^2 = MAG0^2(1/cos^2-1) (3)

    # MAG0^2 + MAG0^2(1/cos^2-1) + MAG2^2 = 1 (1)
    # MAG0^2 * (1/cos^2) + MAG2^2 = 1
    # MAG2^2 = 1 - MAG0^2 * (1/cos^2) (1)

    # MAG0^2 + MAG0^2(1/cos^2-1) = (1/tan^2)*(1-MAG0^2*1/cos^2) (2)
    # MAG0^2 * (1+1/cos^2) = 1/tan^2-MAG0^2*1/tan^2*1/cos^2
    # MAG0^2 * (1+1/cos^2 + 1/(cos^2 * tan^2)) = 1/tan^2
    # MAG0^2 * (cos^2*tan^2+(tan^2 + 1)/(cos^2*tan^2)) = 1/tan^2
    # MAG0^2 = (cos^2 * tan^2) / (tan^2*(tan^2*cos^2 + tan^2+1))

    latMAG = math.degrees(math.atan(tan))
    longMAG = math.degrees(math.acos(cos))
    if MAG[1] <= 0:
        longMAG = 360 - longMAG
    # print("***")
    # print(latMAG)
    # print(longMAG)

    return latMAG, longMAG, tan, cos, MAG, GEO


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

    MAG[0] = sqrt(MAG[0])
    MAG[1] = sqrt(MAG[1])
    MAG[2] = sqrt(MAG[2])
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

    rE = sqrt(GEO[0] * GEO[0] + GEO[1] * GEO[1] + GEO[2] * GEO[2])

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


def main():
    eps = 1e-6
    year = 2016

    for latGEO in range(-175, 175, 7):
        for longGEO in range(-88, 88, 2):
            for altGEO in range(0, 5_000, 500):
                # print(f'{latGEO=}, {longGEO=}')
                _latGEO = math.radians(latGEO)
                _longGEO = math.radians(longGEO)
                _altGEO = altGEO / 1000

                mag1 = geo_2_mag__initial(_latGEO, _longGEO, _altGEO)
                mag2 = geo_2_mag_fixed(_latGEO, _longGEO, _altGEO)

                if abs(mag1[0] - mag2[0]) > eps or abs(mag1[1] - mag2[1]) > eps:
                    print("mag1-mag2", _latGEO, _longGEO, _altGEO, mag1, mag2)

                _latGEO2, _longGEO2, _altGEO2 = mag_to_geo(mag1[0], mag1[1])

                assert abs(_latGEO2 - _latGEO) < eps
                assert abs(_longGEO2 - _longGEO) < eps

                assert abs(math.degrees(_latGEO2) - latGEO) < eps
                assert abs(math.degrees(_longGEO2) - longGEO) < eps

                # print(f'{_latGEO2=}, {_latGEO=}, {_longGEO2=}, {_longGEO=}, {_altGEO2=}, {_altGEO=}, {_latGEO+_latGEO2}, {_longGEO+_longGEO2}')
                # print()
                # print()
                # print()


main()
print("end")
