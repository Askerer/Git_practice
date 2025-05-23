# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 17:57:46 2020

@author: asker
"""

import tensorflow as tf
import pandas as pd

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file("iris_training.csv","https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")

test_path = tf.keras.utils.get_file("iris_test_cvs","https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path,names=CSV_COLUMN_NAMES, header=0)

train.head()

train_y = train.pop('Species')
test_y = test.pop('Species')

#train.shape

def input_fn(features,labels,training=True,batch_size=256):
    
    dataset = tf.data.Dataset.from_tensor_slices((dict(features),labels))
    
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)


my_feature_columns = []

for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

print(my_feature_columns)


classifier = tf.estimator.DNNClassifier(
        feature_columns = my_feature_columns,
        hidden_units=[30,10],
        n_classes=3
        )

classifier.train(input_fn=lambda : input_fn(train,train_y,training=True),steps=7500)


eval_result = classifier.evaluate(
        input_fn=lambda : input_fn(test,test_y,training=False))

print('\n Test set accuracy : {accuracy:0.3f}\n'.format(**eval_result))



def input_pred(features, batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted.")

for feature in features:
    valid = True
    while valid:
        val = input (feature +": ")
        if not val.isdigit(): valid = False
    
    predict[feature] = [float(val)]


predictions = classifier.predict(input_fn = lambda : input_pred(predict))


for pred in predictions:
    class_id = pred['class_ids'][0]
    probability = pred['probabilities'][class_id]
    
    print('Prediction is "{}" ({:.1f}%)'.format(SPECIES[class_id],100*probability))




















