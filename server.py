from glob import glob
import json
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

app = Flask(__name__, static_url_path='')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/images', methods=['GET', 'POST'])
def serve_images_index():
    if request.method == "GET":

        return json.dumps({ 'images': list(map(lambda path: f'/{path}', glob('images/*'))) })

    elif request.method == "POST":

        # with open('upload.txt', 'w') as fp:
        #     fp.write(json.dumps({ 'fedid': request.form['fedidInput'],
        #                           'password': request.form['passwordInput'],
        #                           'image': request.form['imageInput'] }))

        #return redirect('/')

        return json.dumps({ 'fedid': request.form['fedidInput'] })



@app.route('/images/<path:path>')
def serve_images(path):
    return send_from_directory('images', path)
