# Generated by Django 2.2.12 on 2021-02-12 16:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('polls', '0002_auto_20210212_1641'),
    ]

    operations = [
        migrations.AlterField(
            model_name='expences',
            name='date',
            field=models.DateField(),
        ),
    ]
