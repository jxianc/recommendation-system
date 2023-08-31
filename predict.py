import time
import pyspark
import argparse
import json

def predict(train_file, model_file, test_file, res_file, n_weights, sc: pyspark.SparkContext):
    # load train file
    data_rdd = sc.textFile(train_file) \
        .map(lambda x: json.loads(x)) \
        .cache()
    
    # train rdd
    # [user_id, [business_id]]
    train_rdd = data_rdd.map(lambda x: (x['user_id'], x['business_id'])) \
        .groupByKey() \
        .mapValues(lambda x: list(set(x))) \
        .cache()

    # business id to index map
    business_ids = train_rdd.flatMap(lambda x: x[1]).distinct().collect()
    business_index_map = {}
    for i, business_id in enumerate(business_ids):
        business_index_map[business_id] = i

    # user id to index map
    user_ids = train_rdd.map(lambda x: x[0]).distinct().collect()
    user_index_map = {}
    for i, user_id in enumerate(user_ids):
        user_index_map[user_id] = i

    # train dictionary 
    # [(user_index, business_index), stars] 
    train_dict = data_rdd.map(
        lambda x: ((
            user_index_map[x['user_id']], business_index_map[x['business_id']]),
            x['stars']
        )) \
        .distinct() \
        .collectAsMap()
    
    # [user_index, (business_index, star)]
    user_businesses_map = data_rdd.map(lambda x: (
            user_index_map[x['user_id']], 
            (business_index_map[x['business_id']], x['stars'])
        )) \
        .groupByKey() \
        .mapValues(lambda x: list(set(x))) \
        .collectAsMap()
    
    # test rdd
    # [user_id, business_id]
    test_rdd = sc.textFile(test_file) \
        .map(lambda x: json.loads(x)) \
        .map(lambda x: (x['user_id'], x['business_id'])) \
        .cache()

    # model dictionary
    # [(b1_index, b2_index), pearson correlation]
    model_dict = sc.textFile(model_file) \
        .map(lambda x: json.loads(x)) \
        .map(lambda x: ((business_index_map[x['b1']], business_index_map[x['b2']]), x['sim'])) \
        .collectAsMap()

    # test join train
    # [(user_id, predict_business_id), [rated_business_id]]
    joined_rdd = test_rdd.join(train_rdd) \
        .map(lambda x: ((x[0], x[1][0]), x[1][1])) \
        .cache()

    # get pearson correlation model dictionary given a list of pair of business ids
    # -> ((user_id, predict_business_id), [(business_id, sim)])
    def get_sims(x):
        u_id, pb_id = x[0]
        top_n = []
        for b_id in x[1]:
            key = tuple(sorted((pb_id, b_id)))
            key = (business_index_map[key[0]], business_index_map[key[1]])
            if key in model_dict:
                top_n.append((b_id, model_dict[key]))
        return (x[0], sorted(top_n, key=lambda x: x[1], reverse=True)[:n_weights])
            
    # find the pearson correlation for each pair of each user, then sort and take the top N
    # [(user_id, predict_business_id), [(business_id, sim)]]
    predict_rdd = joined_rdd.map(get_sims) \
        .cache()

    def predict(x):
        user_id, predict_business_id = x[0]
        user_index = user_index_map[user_id]

        if len(x[1]) == 0:
            businesses = user_businesses_map[user_index]
            avg = sum(b[1] for b in businesses) / len(businesses)
            return ((user_id, predict_business_id), avg)

        weight_sum = sum(tup[1] for tup in x[1]) # sum over pearson correlation
        num = sum([train_dict[(user_index, business_index_map[x[1][i][0]])] * x[1][i][1] for i in range(len(x[1]))])
        return ((user_id, predict_business_id), num / weight_sum)

    # predict 
    # [(user_id, business_id), predicted_rating]    
    res_rdd = predict_rdd.map(predict).cache()

    # write to json
    res = res_rdd.map(lambda x: {"user_id": x[0][0], "business_id": x[0][1], "stars": x[1]}) \
        .collect()
    with open(res_file, 'w') as file:
        for obj in res:
            json.dump(obj, file)
            file.write('\n')

if __name__ == '__main__':
    start_time = time.time()

    # init pyspark
    sc_conf = pyspark.SparkConf() \
        .setAppName('predict') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '4g') \
        .set('spark.executor.memory', '4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

    # parse arguments
    parser = argparse.ArgumentParser(description='predict')
    parser.add_argument('--train_file', type=str, default='./data/train_review_ratings.json', help='training file')
    parser.add_argument('--model_file', type=str, default='./model.json', help='model file')
    parser.add_argument('--test_file', type=str, default='./data/test_review.json', help='test file')
    parser.add_argument('--res_file', type=str, default='./out.json', help='file to output the result')
    args = parser.parse_args()

    # predict
    n_weights = 20
    predict(args.train_file, args.model_file, args.test_file, args.res_file, n_weights, sc)

    sc.stop()

    # log time
    print('The run time is: ', (time.time() - start_time))
