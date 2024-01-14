# App.py python file for the flask framework
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    input_data = request.form.get('inputField')
    return f'You submitted this data: {input_data}'

if __name__ == '__main__':
    app.run()
