import numpy as np
import pandas as pd
import sys
import yapi

NUM_MAX_VIDEOS = 20

def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    api_key = readfile(sys.argv[3])[0]
    channel_names = readfile(input_file)

    api = yapi.YoutubeAPI(api_key)
    all_videos = [x for x in list(get_all_channel_videos(api, channel_names)) if len(x) > 0]
    all_videos = np.concatenate(all_videos)

    df = pd.DataFrame(all_videos)
    df.columns = ["video_id", "video_title", "description", "tags", "channel", "thumbnail"]

    df.to_csv(output_file, index=False)

def get_all_channel_videos(api, channel_names):
    for channel_name in channel_names:
        print channel_name
        videos = np.array(list(get_channel_videos_info(api, channel_name)))
        yield videos

def get_channel_videos_info(api, channel_name):
    channel_playlist_id = get_channel_playlist_id(api, channel_name)
    channel_videos = get_channel_videos(api, channel_playlist_id)

    for video_id in channel_videos:
        try:
            video = grab_video(api, video_id)
            info = get_video_info(video_id, video)

            yield info
        except AttributeError:
            pass

def readfile(filepath):
    with open(filepath) as f:
        content = [x.strip() for x in f.readlines()]
        return content

def grab_video(api, video_id):
    return api.get_video_info(video_id)

def get_video_info(video_id, video):
    return [x.encode("utf-8") for x in [
        video_id,
        get_video_title(video),
        get_video_description(video).replace("\n", "\\n"),
        #get_video_duration(video),
        str(get_video_tags(video)),
        get_channel_title(video),
        get_video_thumbnail(video)
    ]]

def get_video_title(video):
    return video.items[0].snippet.title

def get_video_description(video):
    return video.items[0].snippet.description

def get_video_duration(video):
    return process_duration(video.items[0].duration.encode("utf-8"))

def process_duration(text_duration):
    text_duration = text_duration.replace("PT", "")

    duration = 0.0
    if "" in text_duration:
        pass
    pass

def get_video_tags(video):
    return [x.encode("utf-8") for x in video.items[0].snippet.tags]

def get_channel_title(video):
    return video.items[0].snippet.channelTitle

def get_video_thumbnail(video):
    return video.items[0].snippet.thumbnails.default.url

def get_channel_id(api, channel_name):
    return api.get_channel_by_name(channel_name).items[0].id

def get_channel_playlist_id(api, channel_name):
    return api.get_channel_by_name(channel_name).items[0].contentDetails.relatedPlaylists.uploads

def get_channel_videos(api, channel_playlist_id):
    return [x.contentDetails.videoId for x in api.get_playlist_items_by_playlist_id(channel_playlist_id, max_results=NUM_MAX_VIDEOS).items]

if __name__ == "__main__":
    main()
