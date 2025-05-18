from django.http import HttpResponse
from django.shortcuts import render
from .forms import HeartDiseaseForm

def home(request):
    context = {
        'name' : "Cindrella Koju"
    }
    return render(request,'homepage.html',context)
    # return HttpResponse("Hello from prediction App")

def heart_disease_form(request):
    form = HeartDiseaseForm()
    return render(request, 'heart_disease_form.html', {'form': form})