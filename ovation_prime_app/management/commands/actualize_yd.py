import datetime

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
import time

from nasaomnireader.omnireader import omni_downloader


def add_month(dt: datetime.date) -> datetime.date:
    if dt.month == 12:
        return datetime.date(dt.year + 1, 1, dt.day)
    return datetime.date(dt.year, dt.month + 1, dt.day)


def add_half_year(dt: datetime.date) -> datetime.date:
    month = dt.month + 6
    new_year = dt.year + (month - 1) // 12
    new_month = month - 12 if month > 12 else month
    return datetime.date(new_year, new_month, dt.day)


def add_year(dt: datetime.date) -> datetime.date:
    return datetime.date(dt.year + 1, dt.month, dt.day)


YA_DISK_TOKEN = getattr(settings, 'YA_DISK_TOKEN', None)
YA_DISK_DIR = getattr(settings, 'YA_DISK_DIR', None)
PROXY_API_URL = getattr(settings, 'PROXY_API_URL', None)
PROXY_API_KEY = getattr(settings, 'PROXY_API_KEY', None)


class Command(BaseCommand):
    help = 'Closes the specified poll for voting'

    steps = {
        'hourly': add_half_year,
        '5min': add_month,
        '1min': add_month,
    }

    cdf_or_txts = [
        'cdf',
        'txt',
    ]

    cadences = [
        'hourly',
        '5min',
        '1min',
    ]

    def add_arguments(self, parser):
        now = datetime.datetime.now()
        month_ago = now - datetime.timedelta(days=35)
        next_month = now + datetime.timedelta(days=35)

        parser.add_argument('--start_dt',  type=datetime.date.fromisoformat, required=False, default=month_ago.date())
        parser.add_argument('--end_dt', type=datetime.date.fromisoformat, required=False, default=next_month.date())
        parser.add_argument('--cdf_or_txt',  type=str, required=False, default=None)

    def _load(self, cdf_or_txt, start_dt, end_dt):
        downloader = omni_downloader(YA_DISK_TOKEN, YA_DISK_DIR, cdf_or_txt=cdf_or_txt, force_download=True)
        for cadence in self.cadences:
            step = self.steps[cadence]

            dt = start_dt
            while dt <= end_dt:
                print(f"download {dt}, {cadence}, {cdf_or_txt}")
                downloader.load_from_nasa_to_yadisk(dt, cadence, proxy_key=PROXY_API_KEY, proxy_url=PROXY_API_URL)

                print(f"downloaded {dt}, {cadence}, {cdf_or_txt}")
                dt = step(dt)
                time.sleep(2)

    def handle(self, *args, **options):
        start_dt = options['start_dt']
        end_dt = options['end_dt']
        cdf_or_txt = options['cdf_or_txt']
        print(start_dt, end_dt, cdf_or_txt)
        if cdf_or_txt is None:
            for cdf_or_txt in self.cdf_or_txts:
                self._load(cdf_or_txt, start_dt, end_dt)
        else:
            self._load(cdf_or_txt, start_dt, end_dt)
