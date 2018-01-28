from __future__ import print_function
from clustering import *
from flask import Flask, request, render_template
from google.cloud import language

import json
import random
import pandas as pd
import yapi

import features
import youtube

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

app = Flask(__name__)

api_key_file = "key.txt"
data_file = "../feature-extractor/output.csv"

api_key = youtube.readfile(api_key_file)[0]
ytapi = yapi.YoutubeAPI(api_key)

client = language.LanguageServiceClient()

data = pd.read_csv(data_file)
model = train_clustering(data)

@app.route("/recommendations")
def recommendations():
    videos_list = request.args.get("previous_videos")
    videos_list = videos_list[1:len(videos_list) - 1]
    videos = [x.encode("utf-8").strip() for x in videos_list.split(",")]

    last_video = videos[0]

    vector = get_video_vector(last_video)
    print(vector)
    recommended_video_ids = predict(data, model, vector)
    s = get_video_info(recommended_video_ids, data)
    random.shuffle(s)
    return str(s)

@app.route("/<video>")
def video_rec(video):
    vector = get_video_vector(video)
    recommended_video_ids = predict(data, model, vector)
    recommended = get_video_info(recommended_video_ids, data)
    random.shuffle(recommended)

    print(recommended)

    return render_template("video.html", videos=recommended)

def get_video_vector(video_id):
    video = youtube.grab_video(ytapi, video_id)
    print(video)
    info = youtube.get_video_info(video_id, video)

    return np.array(features.get_video_info(client, info))[6:]

def get_video_info(video_ids, data):
    entries = []
    for video_id in video_ids:
        entry = data[data["video_id"] == video_id].iloc[0]
        entries.append({
            "video_url": "https://www.youtube.com/watch?v=" + entry["video_id"],
            "thumbnail_url": "https://i.ytimg.com/vi/" + entry["video_id"] + "/hqdefault.jpg",
            "video_title": entry["video_title"]
            })

    return entries

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8800)
