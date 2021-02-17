from django.contrib import admin

# Register your models here.
from .models import Expences
admin.site.site_header = "Expence Prediction Using Linear Regression"


class ExpenceAdmin(admin.ModelAdmin):
    list_display = ("date","Category" , 'income','expences')
    list_filter = ("date","Category","expences")

admin.site.register(Expences,ExpenceAdmin)