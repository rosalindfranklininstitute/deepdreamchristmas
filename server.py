from glob import glob
import json
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

from rq import Queue
from rq.job import Job
from worker import conn

q = Queue(connection=conn)

app = Flask(__name__, static_url_path='')

def process_upload(request):

    with open('upload.txt', 'w') as fp:
        fp.write(json.dumps(request))

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

        print(request)

        job = q.enqueue_call(func=process_upload, args=(request,), result_ttl=5000)
        return json.dumps({ 'id': job.get_id() })


@app.route('/images/<path:path>')
def serve_images(path):
    return send_from_directory('images', path)
