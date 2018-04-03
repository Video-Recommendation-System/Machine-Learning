import numpy as np
import math
import pandas as pd

def test():
    data_raw = pd.read_csv("../../youtube-scraper/output.csv")

    data_raw["description"] = data_raw["description"].astype('str')
    mask = data_raw["description"].str.len() > 1
    data_raw = data_raw.loc[mask]

    data = data_raw.as_matrix(["video_id", "description"])

    wcs = [get_word_counts(d[1]) for d in data]
    data2 = [(data[i][0], wcs[i]) for i in range(0, len(data))]

    reps = kmedoids(40, 10, data2, difference)

    data_raw["cluster"] = [get_cluster(reps, wc, difference) for wc in wcs]

    data_raw.to_csv("results.csv", index=False)

def get_cluster(reps, wc, diffF):
    most_simmilar = None
    cluster = None
    simmilarity = None
    for i in range(0, len(reps)):
        r = reps[i]
        simm = diffF(wc, r)
        if simmilarity is None:
            most_simmilar = r
            simmilarity = simm
            cluster = i
        elif simm < simmilarity:
            most_simmilar = r
            simmilarity = simm
            cluster = i

    return cluster

def difference(wc1, wc2):
    wc1_total = get_total(wc1)
    wc2_total = get_total(wc2)

    total_words = set([w for w in wc1] + [w for w in wc2])

    simmilar_words = 0
    for word in wc1:
        if word in wc2:
            diff = abs(wc1[word] / float(wc1_total) - wc2[word] / float(wc2_total))
            simmilar_words += 1
    
    sum_product = 0
    for w in total_words:
        a = wc1[w] if w in wc1 else 0
        b = wc2[w] if w in wc2 else 0

        sum_product += a * b

    one_product = math.sqrt(sum([v ** 2 for v in wc1.values()]))
    two_product = math.sqrt(sum([v ** 2 for v in wc2.values()]))

    product_sums = one_product * two_product

    return (sum_product / product_sums) + wc1_total / (simmilar_words + 1)

def get_total(wc):
    total = 0
    for word in wc:
        total += wc[word]

    return total

def kmedoids(k, iterations, data, distF):
    """
    data = [(id, word_count)]
    """
    np.random.shuffle(data)
    intial_medoids = [x[1] for x in data[0:k]]

    return kmedoids_iter(iterations, data, intial_medoids, distF)

def kmedoids_iter(iterations_left, data, reps, distF):
    if iterations_left < 1:
        return reps
    else:
        clusters = [[] for _ in reps]
        for (e_id, wc) in data:
            scores = [distF(wc, r_wc) for r_wc in reps]

            most_similar_index = np.argmin(scores)
            clusters[most_similar_index].append((e_id, wc))

        print([len(x) for x in clusters])

        print("-----------")
        for i in range(0, len(reps)):
            c = clusters[i]
            rep = reps[i]

            rep_score = get_total_score(c, rep, distF)
            for e in c:
                e_score = get_total_score(c, e[1], distF)

                if e_score < rep_score:
                    rep = e[1]
                    reps[i] = e[1]

                    rep_score = e_score

            print(rep_score)

        return kmedoids_iter(iterations_left - 1, data, reps, distF)

def get_total_score(cluster, rep, distF):
    total = 0
    for (_, wc) in cluster:
        total += distF(wc, rep)

    return total

def get_word_counts(text):
    word_counts = {}
    words = text.replace("\n", " ").replace("\t", " ").split(" ")
    [increment(w, word_counts) for w in words]

    return word_counts

def increment(word, word_counts):
    if word in word_counts:
        word_counts[word] = word_counts[word] + 1
    else:
        word_counts[word] = 1

if __name__ == "__main__":
    test()
