from django.db import models
Activity = (
    ('food','Food'),
    ('travel', 'Travel'),
    ('clothes','Clothes'),
    ('homeutility','HomeUtility'),
    ('hospital','Hospital'),
)


class Expences(models.Model):
    date = models.DateField(auto_now=False)
    Category = models.CharField(max_length=16, choices=Activity, default='Food')
    income = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    expences = models.DecimalField(max_digits=10, decimal_places=2)
    def __str__(self):
        return self.Category

    
    
