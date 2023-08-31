# Recommendation System

This recommendation system uses an item-based collaborative filtering method. First, build the model by computing the Pearson Correlation of each business pair. Then use this built model to predict the ratings of the user on the business.

**build.py** - build the model command to run this script:

```shell
python build.py –train_file <training dataset> –model_file <file to output the model>
```

eg:

```shell
python build.py –train_file train_review_ratings.json –model_file model.json
```

**predict.py** - predict the ratings command to run this script:

```shell
python predict.py –train_file <training dataset> –model_file <model file> –test_file <test dataset> –res_file <file to output result>
```

eg:

```shell
python predict.py –train_file train_review_ratings.json –model_file model.json –test_file test_review.json –res_file out.json
```

The dataset should be in a specific form

- **training dataset** is a json file where each line is a json object that has three keys, that are _user_id_, _business_id_, and _stars_ (the rating)
- **test dataset** is a json file where each line is a json object that has only two keys, that are _user_id_, and _business_id_
- **model** is a json file that is generated from the built script where each line is a json object that has three keys, that are _b1_, _b2_, and _sim_
  - _b1_ and _b2_ represent a business pair
  - _sim_ represents the Pearson Correlation
