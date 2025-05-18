from django.http import HttpResponse
from django.shortcuts import render

def home(request):
    context = {
        'name' : "Cindrella Koju"
    }
    return render(request,'homepage.html',context)
    # return HttpResponse("Hello from prediction App")