from django.urls import path, include

from . import views
urlpatterns = [
    path('api/v1/get_data/', views.get_ovation_prime_data),
]

