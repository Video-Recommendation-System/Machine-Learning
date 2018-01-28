from __future__ import print_function
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

import pandas as pd
import sys

all_categories = ["/Science/Mathematics","/Science/Physics","/News/Business News","/Home & Garden/Home Appliances","/Science/Earth Sciences","/Reference","/Arts & Entertainment/Music & Audio/Radio","/Science","/Finance/Investing/Currencies & Foreign Exchange","/People & Society/Subcultures & Niche Interests","/Sports/Team Sports","/Reference/Humanities/History","/Health/Public Health","/Law & Government/Public Safety/Law Enforcement","/Sports/Team Sports/Baseball","/Games/Computer & Video Games","/Arts & Entertainment/Fun & Trivia/Flash-Based Entertainment","/Games/Computer & Video Games/Shooter Games","/Home & Garden/Kitchen & Dining","/Computers & Electronics/Software","/Arts & Entertainment/Music & Audio/World Music","/Games","/Sensitive Subjects","/Arts & Entertainment/TV & Video","/Reference/General Reference/Calculators & Reference Tools","/Business & Industrial/Energy & Utilities","/Computers & Electronics/Consumer Electronics","/Games/Roleplaying Games","/Arts & Entertainment","/Arts & Entertainment/Music & Audio/Dance & Electronic Music","/Arts & Entertainment/Music & Audio/Religious Music","/Arts & Entertainment/TV & Video/TV Shows & Programs","/Sports/Team Sports/Cricket","/Jobs & Education/Education/Teaching & Classroom Resources","/Arts & Entertainment/Music & Audio","/Law & Government/Public Safety","/Science/Engineering & Technology","/Food & Drink","/Arts & Entertainment/Performing Arts","/Home & Garden/Kitchen & Dining/Small Kitchen Appliances","/Arts & Entertainment/TV & Video/Online Video","/Business & Industrial","/Law & Government/Public Safety/Crime & Justice","/Arts & Entertainment/Music & Audio/Urban & Hip-Hop","/Computers & Electronics/Software/Multimedia Software","/Arts & Entertainment/Online Media","/Hobbies & Leisure","/Arts & Entertainment/Music & Audio/Music Equipment & Technology","/Arts & Entertainment/Music & Audio/Soundtracks","/Arts & Entertainment/Music & Audio/Rock Music","/People & Society/Religion & Belief","/Computers & Electronics/Software/Business & Productivity Software","/Books & Literature","/Computers & Electronics/CAD & CAM","/Games/Board Games/Chess & Abstract Strategy Games","/Internet & Telecom","/Health/Health Conditions/Infectious Diseases","/Online Communities","/Arts & Entertainment/Comics & Animation/Comics","/Business & Industrial/Energy & Utilities/Renewable & Alternative Energy","/Arts & Entertainment/Comics & Animation/Cartoons","/Law & Government","/Computers & Electronics/Computer Security","/Jobs & Education/Education/Colleges & Universities","/Computers & Electronics","/Science/Biological Sciences","/Science/Computer Science","/News/Sports News","/Science/Earth Sciences/Geology","/Sports","/Arts & Entertainment/Comics & Animation/Anime & Manga","/Arts & Entertainment/Visual Art & Design","/Computers & Electronics/Consumer Electronics/Game Systems & Consoles","/Arts & Entertainment/Music & Audio/Music Reference","/News/Politics","/Business & Industrial/Aerospace & Defense/Space Technology","/Hobbies & Leisure/Special Occasions/Holidays & Seasonal Events","/Arts & Entertainment/Fun & Trivia","/Arts & Entertainment/Humor","/Arts & Entertainment/Movies","/Online Communities/Social Networks","/Arts & Entertainment/Music & Audio/Pop Music","/Home & Garden","/Jobs & Education/Education","/Games/Computer & Video Games/Casual Games","/Pets & Animals/Wildlife","/Business & Industrial/Agriculture & Forestry","/Computers & Electronics/Computer Hardware","/Games/Computer & Video Games/Simulation Games","/Reference/General Reference","/Science/Astronomy","/News","/Arts & Entertainment/Comics & Animation","/Internet & Telecom/Mobile & Wireless/Mobile & Wireless Accessories"]

def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    data = pd.read_csv(input_file)

    client = language.LanguageServiceClient()

    sentiments = []
    classifications = []
    descriptions = data["description"]
    titles = data["video_title"]
    unique_classifications = set()
    for x in range(0, len(descriptions)):
        text = descriptions.iloc[x]
        print(titles[x])

        document = types.Document(
            content = text,
            type = enums.Document.Type.PLAIN_TEXT)
        sentiments.append(get_sentiment(client, document))
        categories = []
        for category in get_classification(client, document):
            categories.append((category.name.encode("utf-8"), category.confidence))
            unique_classifications.add(category.name.encode("utf-8"))

        classifications.append(categories)

    data["sentiment"] = sentiments
    data["classifications"] = classifications

    for classification in unique_classifications:
        column = []
        for x in range(0, len(data["classifications"])):
            categories_set = data["classifications"].iloc[x]
            any_match = False
            for category_pair in categories_set:
                if classification in category_pair[0]:
                    if len(column) == x + 1:
                        column[x] = category_pair[1]
                    else:
                        column.append(category_pair[1])
                        any_match = True
            if not any_match:
                column.append(0.0)

        data[classification] = column

    print(data)
    data.to_csv(output_file, index=False)

def get_video_info(client, video_info):
    document = types.Document(
        content = video_info[2],
        type = enums.Document.Type.PLAIN_TEXT)

    sentiment = get_sentiment(client, document)
    video_info.append(sentiment)

    categories = []
    for category in get_classification(client, document):
        categories.append((category.name.encode("utf-8"), category.confidence))

    for classification in all_categories:
        value = None
        any_match = False

        for category_pair in categories:
            if classification in category_pair[0]:
                if value is not None:
                    value = category_pair[1]
                else:
                    value = category_pair[1]
                    any_match = True
        if not any_match:
            value = 0.0

        video_info.append(value)

    return video_info

def get_sentiment(client, document):
    return client.analyze_sentiment(document=document).document_sentiment.magnitude

def get_classification(client, document):
    try:
        return client.classify_text(document).categories
    except Exception:
        return []

if __name__ == "__main__":
    main()
