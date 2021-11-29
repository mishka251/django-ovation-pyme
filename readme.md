# django-ovation-pyme
Web API implementation form OvationPrime model with python and django

Установка
```bash
pip install -e git+https://github.com/mishka251/django-ovation-pyme.git@{VERSION}#egg=django-ovation-prime
```

(на примере версии 0.1.1)
```bash
pip install -e git+https://github.com/mishka251/django-ovation-pyme.git@0.1.1#egg=django-ovation-prime
```

Подключение к проекту:
1. Добавить в `settings.py` в `INSTALLED_APPS` значение `ovation_prime_app`
2. Добавить в `settings.py` настройки из примеров `ovation_prime_site/base_settings.py` `ovation_prime_site/settings_template.py`
3. Добавить в `urls.py` подключение url из приложения `include(ovation_prime_app.urls)`

Доступные методы API в текущей версии(развернуто на проекте geotest `http://gimslaw8.bget.ru/ovation_prime`)

1. `api/v1/conductance_interpolated/` - ??? Параметры dt - датавремя, _type - тип hallgrid/pedgrid
2. `api/v1/conductance/` - ??? Параметры dt - датавремя, _type - тип hallgrid/pedgrid
3. `api/v1/weighted_flux/` - ??? Параметры   dt - датавремя в формате `yyyy-mm-ddTHH:MM:SS`, atype  'diff'/'mono'/'wave'/'ions'  - type of aurora for which to load regression coeffients, jtype 'energy'/'number' Type of flux you want to estimate
4. `api/v1/seasonal_flux/` - ???  Параметры 
  - dt - датавремя в формате `yyyy-mm-ddTHH:MM:SS`,
  - atype 'diff'/'mono'/'wave'/'ions',  type of aurora for which to load regression coeffients
  - jtype'energy'/'number' Type of flux you want to estimate
  - seasonN 'winter'/'summer'/'spring'/'fall'
  - seasonS 'winter'/'summer'/'spring'/'fall'