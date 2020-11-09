from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('knn.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['fir']
    data2 = request.form['sec']
    data3 = request.form['thr']
    data4 = request.form['fou']
    data5 = request.form['fif']
    data6 = request.form['six']
    data7 = request.form['sev']
    data8 = request.form['eig']
    arr = np.array([[data1, data2, data3, data4,data5,data6,data7,data8]])
    pred = model.predict(arr)
    return render_template('predict.html', data=pred)


if __name__ == "__main__":
    app.run(host='127.0.0.1',port=8000,debug=True)