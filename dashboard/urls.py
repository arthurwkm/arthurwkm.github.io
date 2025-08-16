from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.dashboard_home, name='home'),
    path('results/', views.model_results, name='results'),
    path('visualizations/', views.visualizations, name='visualizations'),
    path('predict/', views.predict_mental_state, name='predict'),
    path('about/', views.about, name='about'),
    path('learn/', views.learn, name='learn'),
]