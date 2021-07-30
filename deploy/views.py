from django.shortcuts import render
import pickle
import pandas as pd

car = pd.read_csv('quikr_car_cleaned.csv')

def home(request):
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    return render(request, "index.html", {'companies': companies,
                                          'car_models': car_models,
                                          'years': year,
                                          'fuel_types': fuel_type})


def result(request):

    model = pickle.load(open('CarPricePredictionLinearRegressionModel.pkl', 'rb'))

    name = request.GET['car_models']
    company = request.GET['company']
    year = int(request.GET['year'])
    kms = int(request.GET['kilo_driven'])
    fuel_type = request.GET['fuel_type']

    ans = model.predict(pd.DataFrame([[name, company, year, kms, fuel_type]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))


    return render(request, "result.html", {'ans': ans[0]})
