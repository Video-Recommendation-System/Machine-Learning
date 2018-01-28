from clustering import *
from flask import Flask, request

import pandas as pd

app = Flask(__name__)

data_file = "csv2.csv"
data = pd.read_csv(data_file)
model = train_clustering(data)

@app.route("/recommendations")
def recommendations():
    videos_list = request.args.get("previous_videos")
    videos_list = videos_list[1:len(videos_list) - 1]
    videos = [x.encode("utf-8").strip() for x in videos_list.split(",")]

    last_video = videos[0]

    vector = get_video_vector(vector)
    recommended_video_ids = predict(model, vector)
    return get_video_info(recommended_video_ids, data)
