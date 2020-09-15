from flask import Flask
from flask import request
from flask import send_file
import json
import os
import subprocess


app = Flask(__name__)


@app.route('/programs')
def programs():
    files =  os.listdir('.')
    programs = list(filter(lambda filename: '.exe' in filename, files))
    programs = list(map(lambda program: program.replace('.exe', ''), programs ))
    return json.dumps(programs)



@app.route('/programs/<name>')
def program(name):
    name += '.json'
    with open(name) as f:
        jsonData = json.load(f)
    return json.dumps(jsonData)


@app.route('/exec/<program>', methods = ['POST'])
def exec(program):
    params = request.get_json()
    print(params)
    print(program)
    print(os.popen(f'mpirun -np {params["np"]} {program}.exe {params["input"]}').read())
    return send_file(params["output"])


@app.route('/upload', methods = ['POST'])
def uploadInputs():
    for file in request.files.getlist('file'):
        file.save(file.filename)
    return 'ok'

if __name__ == '__main__':
    app.run(debug=True, port = 5000)