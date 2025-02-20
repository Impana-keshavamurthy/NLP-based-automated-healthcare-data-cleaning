from django.urls import path
from .views import clean_data, index, generate_boxplots

urlpatterns = [
    path('', index, name='index'),
    path('clean-data/', clean_data, name='clean_data'),
    path('generate-boxplot/', generate_boxplots, name='generate_boxplot'),  # Add this line
]

