from __future__ import print_function
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

import pandas as pd
import sys

all_categories = ["/Science/Computer Science","/Jobs & Education/Education","/Science/Mathematics","/Arts & Entertainment/TV & Video/Online Video","/Arts & Entertainment/Music & Audio/World Music","/Arts & Entertainment/Music & Audio/Music Streams & Downloads","/Games","/Science/Physics","/Arts & Entertainment/Music & Audio/Urban & Hip-Hop","/Sports","/Food & Drink/Cooking & Recipes","/Arts & Entertainment/Comics & Animation/Anime & Manga","/Computers & Electronics/Software/Multimedia Software","/Arts & Entertainment/TV & Video","/Computers & Electronics/Consumer Electronics/Game Systems & Consoles","/Arts & Entertainment/Online Media","/Hobbies & Leisure","/News/Sports News","/Computers & Electronics/Computer Hardware/Computer Drives & Storage","/Home & Garden/Home Appliances","/Reference/General Reference/Calculators & Reference Tools","/Arts & Entertainment/Music & Audio/Music Reference","/Arts & Entertainment/Music & Audio/Soundtracks","/Computers & Electronics/Software","/Arts & Entertainment/Music & Audio/Rock Music","/Hobbies & Leisure/Special Occasions/Holidays & Seasonal Events","/Arts & Entertainment/Music & Audio/Music Equipment & Technology","/Games/Roleplaying Games","/Arts & Entertainment","/Arts & Entertainment/Music & Audio/Dance & Electronic Music","/Adult","/Internet & Telecom","/Business & Industrial/Agriculture & Forestry","/Science/Earth Sciences","/Autos & Vehicles","/News/Politics","/People & Society/Religion & Belief","/Reference","/Arts & Entertainment/TV & Video/TV Shows & Programs","/Home & Garden/Kitchen & Dining","/Sports/Team Sports/Cricket","/Shopping/Apparel","/Arts & Entertainment/Humor","/Arts & Entertainment/Movies","/Arts & Entertainment/Music & Audio/Radio","/Sports/Individual Sports/Cycling","/Business & Industrial/Business Operations/Management","/Computers & Electronics/Programming","/Arts & Entertainment/Fun & Trivia","/Health/Health Conditions/Infectious Diseases","/Arts & Entertainment/Music & Audio/Pop Music","/Arts & Entertainment/Music & Audio/Jazz & Blues","/Science","/Home & Garden","/Online Communities","/Computers & Electronics/Consumer Electronics","/Arts & Entertainment/Comics & Animation/Comics","/Games/Computer & Video Games/Casual Games","/Finance/Investing/Currencies & Foreign Exchange","/Arts & Entertainment/Music & Audio/Classical Music","/Arts & Entertainment/Music & Audio","/People & Society/Subcultures & Niche Interests","/Books & Literature","/Computers & Electronics/Computer Hardware","/Reference/General Reference","/Law & Government","/People & Society/Social Sciences/Psychology","/Reference/Humanities/History","/Arts & Entertainment/Comics & Animation/Cartoons","/Science/Biological Sciences/Neuroscience","/Online Communities/Blogging Resources & Services","/Science/Astronomy","/Home & Garden/Kitchen & Dining/Small Kitchen Appliances","/Arts & Entertainment/Comics & Animation","/Sports/Team Sports/Baseball","/Games/Computer & Video Games","/Science/Mathematics/Statistics","/Jobs & Education/Education/Colleges & Universities","/Computers & Electronics","/Arts & Entertainment/Fun & Trivia/Flash-Based Entertainment","/Science/Biological Sciences","/Games/Online Games/Massively Multiplayer Games"
]

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
