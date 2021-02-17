from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from .views import add_data,delete,send_data,predict,train,predict_data

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('add/', add_data, name='add'),
    path('delete/<pk>', delete, name='delete'),
    path('api/data/', send_data, name='send_data'),
    path('predict/', predict, name='predict'),
    path('predict_data/', predict_data, name='predict_data'),
    path('train/', train, name='train'),
]+ static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)