from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, JsonResponse
from .models import Expences
from django.contrib import messages
from django.db.models import Sum
import numpy as np
from django.db.models.functions import TruncDate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import datetime as dt
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
# from sklearn.preprocessing import 
from  datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import json

# Create your views here.

def add_data(request):
    if request.method == "POST":
        date = request.POST['date']
        activity = request.POST['activity']
        print(activity)
        expences = request.POST['expence']
        inst = Expences(date=date,Category=activity,expences=expences)
        inst.save()
        messages.success(request, "Data succesfully Added !")
    context = Expences.objects.all().order_by('date')[:10]  
    data = {
        'context' : context
    }  
    return render(request,'add_data.html',data)

def index(request):
    total_expences = Expences.objects.all().aggregate(Sum('expences'))
    total_income = Expences.objects.all().aggregate(Sum('income'))
    expences = {
        'total_expences' : total_expences,
        'total_income'  : total_income

    }  
    return render(request,'index.html',expences)

def delete(request,pk):
    data = get_object_or_404(Expences,id=pk)
    data.delete()
    messages.warning(request, "Data Delete succesfully !")
    return redirect('add')
    

def  send_data(request, *arg,**kwarg):
    # retrive data 
    date = Expences.objects.values_list('date').annotate(Sum('expences'))
    # seperate date and expences into different list
    date_expences_array =np.array(date)
    date_list = date_expences_array[:,0]
    expences_list = date_expences_array[:,1]
    activity = Expences.objects.values_list('Category', flat=True)
    expences = Expences.objects.values_list('expences', flat=True)
    # get expences with category
    expences_with_activity = Expences.objects.values_list('Category').annotate(Sum('expences'))
    cat_expences_array =np.array(expences_with_activity)
    cat_list = cat_expences_array[:,0]
    expences_list_cat = cat_expences_array[:,1]

    #expences with year
    expences_with_year = Expences.objects.values_list('date__year').annotate(Sum('expences'))
    expences_with_year =np.array(expences_with_year)
    year_list = expences_with_year[:,0]
    year_list_expences = expences_with_year[:,1]


    #daily income 

    expences_with_year = Expences.objects.values_list('date__year').annotate(Sum('expences'))
    expences_with_year =np.array(expences_with_year)
    year_list = expences_with_year[:,0]
    year_list_expences = expences_with_year[:,1]


    # category_with expence = income_posts.values("category__name").distinct().annotate(total=Sum("amount"))
    data = {
        'date_list' : list(date_list),
        'expences_list' : list(expences_list),
        'cat_list' : list(cat_list),
        'expences_list_cat' : list(expences_list_cat),
        'year_list' : list(year_list),
        'year_list_expences' : list(year_list_expences),
        
    }
    return JsonResponse(data)
def predict_data(request,*arg, **kwarg):
    if request.method == "GET":
        date = request.GET["date"]
        date_time = datetime.fromisoformat(date)
        date_int = 10000*date_time.year + 100*date_time.month + date_time.day
        category = request.GET["category"]
        if category == "food":
            cat = 2
        elif category == "travel":
            cat = 4
        elif category == "clothes":
            cat = 1
        elif category == "HomeUtility":
            cat = 3
        else:
            cat = 2
    with open('classify_data.pickle', 'rb') as pickle_saved_data:
        unpickled_data = pickle.load(pickle_saved_data)
    
    predict_result_array = unpickled_data.predict([[date_int,cat]])
    predict_data =  float(predict_result_array[0][0])
    print("predicted data is {}".format(predict_data))
    
    
    return JsonResponse({"prediction_result": predict_data})



    # return JsonResponse({"data":"data"})
def predict(request,*arg, **kwarg):
    # data_df = list(Expences.object.all())
    return render(request,'predict.html')
def train(request,*arg,**kwarg):
    date_array = []
    expences_array_data = []
    labelencoder = LabelEncoder()
    ordinalencoder = OrdinalEncoder()
    date_df = Expences.objects.values_list('date', flat=True)
    for date in date_df:
        date_int = 10000*date.year + 100*date.month + date.day
        date_array.append(date_int)
    # print(date_array)
        


    category_df = Expences.objects.values_list('Category', flat=True)
    category_array = np.array(category_df)
    encoded_category = labelencoder.fit_transform(category_array)
    # print(encoded_category)


    expences_df = Expences.objects.values_list('expences', flat=True)
    expences_array = np.array([expences_df])

    print(expences_array)


    X = arr = np.stack((date_array, encoded_category))
    x_data = np.transpose(X)
    y = expences_array
    y_data = y.reshape(-1,1)
    # print(x_data, y_data[0:5])

    X_train, X_test, y_train, y_test = train_test_split(x_data,y_data, test_size=0.3, random_state=42)

    # #Fitting Linear Regression to the training set
    multiple_reg = LinearRegression()
    TrainData = multiple_reg.fit(X_train, y_train)
        
    classifier_data = open("classify_data.pickle", "wb")
    pickle.dump(TrainData, classifier_data)
    classifier_data.close()


    # #predicting the Test set result
    y_pred = multiple_reg.predict(X_test)

    #accuracy 
    acc = r2_score(y_test, y_pred)
    print(acc)
    data = {

        "accuracy":acc,

        }
    return JsonResponse(data)


    
    # return render(request,'predict.html')
