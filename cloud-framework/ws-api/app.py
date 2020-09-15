from flask import Flask, request, jsonify, make_response, send_file
from werkzeug.security import generate_password_hash, check_password_hash
import uuid
import datetime
import requests
import json
from functools import wraps

from flask_jwt_extended import (create_access_token, create_refresh_token, jwt_required)
import jwt
import bcrypt
from PIL import Image
import io


app = Flask(__name__)

app.config['SECRET_KEY'] ='secretkey'

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.args.get('token')

        if not token:
            return jsonify({'message': 'No hay token'}), 403

        try:
            data = jwt.decode(token, app.config['SECRET_KEY'])
        except:
            return jsonify({'message': 'token invalido'}), 403

        return f(*args, **kwargs)

    return decorated


@app.route('/login')
def login():
    auth = request.authorization
    
    if auth and auth.password == 'password':
        token = jwt.encode({'user': auth.username, 'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes = 30)}, app.config['SECRET_KEY'])
        return jsonify({'token': token.decode('UTF-8')})
    return make_response('could no verify', 401, {'WWW-Authenticate' : 'Basic realm="LoginRequired"'})



@app.route('/programs')
@token_required
def apps():
    r = requests.get('http://localhost:5000/programs')
    data = r.text
    return data


@app.route('/programs/<name>')
@token_required
def description(name):
    r = requests.get('http://localhost:5000/programs/' + name)
    data = r.text
    return data


@app.route('/upload', methods = ['POST'])
@token_required
def upload():
    for f in request.files.getlist('file'):
        send = {'file':(f.filename, f)}
        response = requests.post('http://localhost:5000/upload', files = send)
    return response.text



@app.route('/exec/<name>', methods = ['POST'])
@token_required
def exec(name):
    params = request.get_json()
    r = requests.post('http://localhost:5000/exec/' + name, json = params)
    img = Image.open(io.BytesIO(r.content))
    img.save('output.jpg')
    return send_file('output.jpg')


app.secret_key = 'mysecret'

if __name__ == '__main__':
    app.run(debug = True, port = 5001)