from training_model import TweetRecognition
from app import app
from flask import request, abort, jsonify

@app.route('/', methods=['POST'])

def process():
    if not request.json or not "text" in request.json:
        abort(400)
    text = request.json['text']
    tweet = TweetRecognition()
    score = tweet.get_predict(text)
    if (score > 2):
        answer = "positive"
    elif (score == -1):
        answer = "unknown"
    else:
        answer = "negative"
    responses = jsonify({"result":answer})
    responses.status_code = 200
    return (responses)
