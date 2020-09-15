from flask import Flask
from flask import request, render_template
from flask_socketio import SocketIO

app = Flask('__name__')
app.config['SECRET_KEY'] = 'secret!'

socketio = SocketIO(app)

@app.route('/updateData', methods = ['POST'])
def main():
    data = request.json
    socketio.emit('data', data)
    return 'ok'


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, port = 5001)