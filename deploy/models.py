from django.db import models

# Create your models here.

class Humbu(models.Model):

    NDOU = models.CharField(max_length=100)
    CLIFF = models.FloatField()

    def __str__(self):

        return self.NDOU