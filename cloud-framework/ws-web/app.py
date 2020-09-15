from flask import Flask
from flask import request
from flask import render_template
from flask import session
from flask import send_file
from flask import Response
from flask import make_response
from flask import url_for
from flask import redirect
import requests
import json
from PIL import Image
import io
from flask_login import logout_user, LoginManager
from flask_pymongo import PyMongo
import bcrypt


app = Flask(__name__)

app.config['MONGO_DBNAME'] = 'educaweb'
app.config['MONGO_URI'] = 'mongodb+srv://alex:1234@cluster0.iklun.gcp.mongodb.net/educaweb?retryWrites=true&w=majority'

mongo = PyMongo(app)
app.secret_key = 'secret'


@app.route('/')
def index():
    if 'username' in session:
        r = requests.get('http://localhost:5000/programs')
        data = r.text
        return render_template('index.html', programs = json.loads(data))
    else:
        return render_template('login.html')


@app.route('/login', methods = ['POST'])
def authorize():
    users = mongo.db.users
    login_user = users.find_one({'name': request.form['username']})
    if login_user:
        if bcrypt.hashpw(request.form['password'].encode('utf-8'), login_user['password'].encode('utf-8')) == login_user['password'].encode('utf-8'):
            session['username'] = request.form['username']
            return redirect(url_for('index'))
    return 'Invalid username or password'



@app.route('/uploadProgram')
def uploadProgram():
    return render_template('upload.html')



@app.route('/form/<program>')
def form(program):
    r = requests.get('http://localhost:5000/programs/' + program)
    data = json.loads(r.text)
    return render_template('form.html', program = data)


@app.route('/exec/<program>', methods = ['POST'])
def exec(program):
    data = {
        'input': 'test.jpg',
        'output': request.form['output-img'],
        'np': request.form['np']
    }

    print(request.files)
    for file in request.files.getlist('file'):
        file.save(file.filename)
        send = {'file':open(file.filename, 'rb')}
        response = requests.post('http://localhost:5000/upload', files = send)
        response = requests.post('http://localhost:5000/exec/' + program, json = data)
        img = Image.open(io.BytesIO(response.content))
        img.save('static/temp/output.png', 'png')
        return send_file('static/temp/output.png')
    return 'ok'


@app.route('/upload', methods = ['POST'])
def upload():
    for file in request.files.getlist('file'):
        file.save(file.filename)
        send = {'file':open(file.filename, 'rb')}
        response = requests.post('http://localhost:5000/upload', files = send)
    return 'ok'


if __name__ == '__main__':
    app.run(debug=True, port=3000)
