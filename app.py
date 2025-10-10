import pickle
import numpy as np
# from numpy import multiarray
# from numpy import _multiarray_umath, overrides
from flask import Flask,render_template,request
app=Flask(__name__)
scaler=pickle.load(open('scaler.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
class_names=['Lawyer ( CLAT Examination Preparation Course )', 'Doctor ( Neet Preparation Course )', 'Government Officer ( SSC Examination Preparation Course )', 'Artist ( Visual Artist Course )', 'Unknown',
       'Software Engineer ( JEE Preparation course , Dsa )', 'Teacher ( TET Examination Preparation Course )', 'Business Owner', 'Scientist',
       'Banker ( Banking Examination Preparation Course )', 'Writer', 'Accountant ( Banking Examination Preparation Course )', 'Designer ( Visual Artist Course )',
       'Construction Engineer ( Construction Site Engineer Course )', 'Game Developer', 'Stock Investor',
       'Real Estate Developer']
def Recommendation(gender,part_time_job,absence_days,extracurricular_activities,weekly_self_study_hours,
                   math_score,history_score,physics_score,chemistry_score,biology_score,
                   english_score,geography_score,total_score,average_score):
    gender_encoded=1 if gender.lower()=='female' else 0
    part_time_job_encoded=1 if part_time_job else 0
    extracurricular_activities_encoded=1 if extracurricular_activities else 0
    feature_array=np.array([[gender_encoded,part_time_job_encoded,absence_days,extracurricular_activities_encoded,weekly_self_study_hours,
                             math_score,history_score,physics_score,chemistry_score,biology_score,english_score,geography_score,total_score,
                             average_score]])
    scaled_features=scaler.transform(feature_array)
    probabilities=model.predict_proba(scaled_features)
    top_classes_idx=np.argsort(-probabilities[0])[:5]
    top_classes_names_probs=[(class_names[idx],probabilities[0][idx]) for idx in top_classes_idx]
    # return probabilities
    return top_classes_names_probs
@app.route('/')
def home():
    return render_template("home.html")
@app.route('/recommend')
def recommend():
    return render_template("recommend.html")
@app.route('/pred',methods=['POST','GET'])
def pred():
    if request.method=='POST':
        gender=request.form['gender']
        part_time_job=request.form['part_time_job'] =='yes'
        absence_days=int(request.form['absence_days'])
        extracurricular_activities=request.form['extracurricular_activities'] =='true'
        weekly_self_study_hours=int(request.form['weekly_self_study_hours'])
        math_score=int(request.form['math_score'])
        history_score=int(request.form['history_score'])
        physics_score=int(request.form['physics_score'])
        chemistry_score=int(request.form['chemistry_score'])
        biology_score=int(request.form['biology_score'])
        english_score=int(request.form['english_score'])
        geography_score=int(request.form['geography_score'])
        total_score=float(request.form['total_score'])
        average_score=float(request.form['average_score'])
        recommendations=Recommendation(gender, part_time_job, absence_days, extracurricular_activities,
                                        weekly_self_study_hours, math_score, history_score, physics_score,
                                        chemistry_score, biology_score, english_score, geography_score, total_score, average_score)
    #     final_recommendation=Recommendation(gender='female',
    #                                 part_time_job=False,
    #                                 absence_days=2,
    #                                 extracurricular_activities=False,
    #                                 weekly_self_study_hours=7,
    #                                 math_score=100,
    #                                 history_score=80,
    #                                 physics_score=90,
    #                                 chemistry_score=90,
    #                                 biology_score=70,
    #                                 english_score=60,
    #                                 geography_score=60,
    #                                 total_score=520,
    #                                 average_score=74.285714)
    # print("Top recommended studies with probabilities:")
    # print("-"*40)
    # for class_name , probability in final_recommendation:
    #     print(f"{class_name} with probablity {probability}")
        return render_template('result.html',recommendations=recommendations)
    return render_template('home.html')
if __name__=='__main__':
    app.run(debug=True)  