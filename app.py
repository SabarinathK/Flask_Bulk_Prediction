from flask import Flask, render_template, request,make_response
import pandas as pd
import joblib as jb
from label import encode

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def prediction():
    if request.method=='POST':
        file=request.files['file']
        df=pd.DataFrame(pd.read_csv(file))
        data=encode(df)
        model=jb.load('mushroom.pkl')
        result=pd.DataFrame(model.predict(data))
        result.columns=['prediction_result']
        result=result.replace({1:'Edible' , 0:'Poisons'})
        response=make_response(result.to_csv())
        response.headers["Content-Disposition"] = "attachment; filename=Prediction.csv"     
        response.mimetype='text/CSV'
        return response
        
if __name__ == ('__main__'):
    app.run(debug=True)