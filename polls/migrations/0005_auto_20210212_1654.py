# Generated by Django 2.2.12 on 2021-02-12 16:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('polls', '0004_auto_20210212_1648'),
    ]

    operations = [
        migrations.AlterField(
            model_name='expences',
            name='expences',
            field=models.DecimalField(decimal_places=5, max_digits=10),
        ),
    ]
