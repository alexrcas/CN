from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import Row,SQLContext

import sys
import requests
from pyspark.sql.types import *

import pandas as pd
import json

def aggregate_tags_count(new_values, total_sum):
    return sum(new_values) + (total_sum or 0)

def get_sql_context_instance(spark_context):
    if ('sqlContextSingletonInstance' not in globals()):
            globals()['sqlContextSingletonInstance'] = SQLContext(spark_context)
    return globals()['sqlContextSingletonInstance']


def sendData(df):
    df = df.toPandas()
    data = df.values
    values = []
    for item in data.tolist():
        pair = {'word': item[0], 'count': item[1]}
        values.append(pair)

    print(values)
    url = 'http://localhost:5001/updateData'
    r = requests.post(url, json=json.dumps(values))
    


def process_rdd(time, rdd):
    print("----------- %s -----------" % str(time))
    try:
        sql_context = get_sql_context_instance(rdd.context)
        print("Get spark sql singleton context from the current context ----------- %s -----------" % str(time))
        row_rdd = rdd.map(lambda w: Row(word=w[0], word_count=w[1]))
        hashtags_df = sql_context.createDataFrame(row_rdd)

        sql_context.registerDataFrameAsTable(hashtags_df, "hashtags")
        
        hashtag_counts_df = sql_context.sql("select word , word_count from hashtags order by word_count desc limit 20")
        sendData(hashtag_counts_df)
        hashtag_counts_df.coalesce(1).write.format('com.databricks.spark.csv').mode('overwrite').option("header", "true").csv("/home/sphashtag_file.csv") 
           
    except Exception as e:
        print(e)



conf = SparkConf()
conf.setAppName("TwitterStreamApp")

sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

ssc = StreamingContext(sc, 2)

ssc.checkpoint("checkpoint_TwitterApp")

dataStream = ssc.socketTextStream("localhost",5556)

words = dataStream.flatMap(lambda line: line.split(' '))

hashtags = words.filter(lambda w: ('#' in w)).map(lambda x : (x, 1))

tags_totals = hashtags.updateStateByKey(aggregate_tags_count)

tags_totals.foreachRDD(process_rdd)

ssc.start()

ssc.awaitTermination()