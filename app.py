from flask import Flask, render_template, request,jsonify
import pandas as pd
from src.prediction import translator

app = Flask(__name__)  # initialising the flask

@app.route('/', methods=['GET', 'POST']) # To render Homepage
def home_page():
    return render_template('index.html')


@app.route('/scrap', methods=['GET','POST']) # route with allowed methods as POST and GET
def index():
    if request.method == 'POST':
        tr=translator.get_translation(request.form['content'])
        print(tr)
        return render_template('results.html', tr=tr[:-5])  # showing the review to the user



if __name__ == "__main__":
    app.run(debug=True)