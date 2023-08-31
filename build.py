import time
import pyspark
import argparse
import json
import math
from itertools import combinations

def build(train_file, model_file, co_rated_thr, sc: pyspark.SparkContext):    
    # load train file into a rdd
    train_rdd = sc.textFile(train_file) \
        .map(lambda x: json.loads(x)) \
        .cache()
    
    # (user_id, [business_id])
    user_business_rdd = train_rdd.map(lambda x: (x['user_id'], x['business_id'])) \
        .groupByKey() \
        .mapValues(lambda x: list(set(x))) \
        .cache()

    # index to business id map
    business_ids = user_business_rdd.flatMap(lambda x: x[1]).distinct().collect()
    business_index_map = {}
    for i, business_id in enumerate(business_ids):
        business_index_map[business_id] = i

    # index to user id map
    user_ids = user_business_rdd.map(lambda x: x[0]).distinct().collect()
    user_index_map = {}
    for i, user_id in enumerate(user_ids):
        user_index_map[user_id] = i

    # [(user_index, business_index), stars]
    train_dict = train_rdd.map(lambda x: (
            (user_index_map[x['user_id']], business_index_map[x['business_id']]), 
            x['stars']
        )) \
        .distinct() \
        .collectAsMap()

    # [business_pair, [user_index]]
    rdd = user_business_rdd.mapValues(lambda x: [tuple(sorted(pair)) for pair in combinations(x, 2)]) \
        .flatMap(lambda x: [(pair, user_index_map[x[0]]) for pair in x[1]]) \
        .groupByKey() \
        .filter(lambda x: len(x[1]) >= co_rated_thr) \
        .cache()

    # compute pearson correlation
    def pearson_correlation(x):
        pair, user_indices = x
        b1, b2 = pair
        b1_index = business_index_map[b1]
        b2_index = business_index_map[b2]
        r1 = []
        r2 = []
        for user_index in user_indices:
            r1.append(train_dict[(user_index, b1_index)])
            r2.append(train_dict[(user_index, b2_index)])

        n = len(r1)
        avg1 = sum(r1) / n
        avg2 = sum(r2) / n

        diff1 = [r-avg1 for r in r1]
        diff2 = [r-avg2 for r in r2]
        num = sum([diff1[i] * diff2[i] for i in range(n)])
        denom = math.sqrt(sum([diff1[i] ** 2 for i in range(n)])) * \
                math.sqrt(sum([diff2[i] ** 2 for i in range(n)]))

        if denom == 0:
            return (pair, 0)
        
        # (pair, sim)
        return (pair, round(num / denom, 7))

    # (pair, sim)
    model_rdd = rdd.map(pearson_correlation) \
        .filter(lambda x: x[1] > 0.1) \
        .cache()

    model = model_rdd.map(lambda x: {"b1": x[0][0], "b2": x[0][1], "sim": x[1]}) \
        .collect()

    # output model    
    with open(model_file, 'w') as file:
        for obj in model:
            json.dump(obj, file)
            file.write('\n')

if __name__ == '__main__':
    start_time = time.time()

    # init pyspark
    sc_conf = pyspark.SparkConf() \
        .setAppName('build') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '4g') \
        .set('spark.executor.memory', '4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

    # parse arguments
    parser = argparse.ArgumentParser(description='build a model')
    parser.add_argument('--train_file', type=str, default='./data/train_review_ratings.json', help='training file')
    parser.add_argument('--model_file', type=str, default='./model.json', help='file to output the built model')
    args = parser.parse_args()

    # build model
    co_rated_thr = 3
    build(args.train_file, args.model_file, co_rated_thr, sc)

    sc.stop()

    # log time
    print('The run time is: ', (time.time() - start_time))
