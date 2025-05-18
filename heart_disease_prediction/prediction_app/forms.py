from django import forms

class HeartDiseaseForm(forms.Form):
    age = forms.IntegerField(label="Age")
    sex = forms.ChoiceField(choices=[(0, 'Female'), (1, 'Male')], label="Sex")
    blood_pressure = forms.IntegerField(label="Blood Pressure")
    cholesterol = forms.IntegerField(label="Cholesterol Level")
    blood_sugar = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')], label="Blood Sugar")
    electrocardiographic_result = forms.ChoiceField(choices=[(0, 'Normal'), (1, 'Abnormal')], label="Electrocardiographic Result")
    max_heart_rate = forms.IntegerField(label="Maximum Heart Rate")
    exercise_angina = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')], label="Exercise Angina")
    oldpeak = forms.FloatField(label="Oldpeak")
    slope = forms.ChoiceField(choices=[(0, 'Up'), (1, 'Flat'), (2, 'Down')], label="Slope")
