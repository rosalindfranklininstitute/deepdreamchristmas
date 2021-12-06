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

@app.route('/images')
def serve_images_index():
    return json.dumps({ 'images': list(glob('images/*')) })

@app.route('/images/<path:path>')
def serve_images(path):
    return send_from_directory('images', path)
