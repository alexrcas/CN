# Práctica 8: Spark
* Alexis Rodríguez Casañas

## Descripción del problema planteado
El reto consiste en utilizar Spark para contar los hashtags más mencionados en Twitter. Con el tiempo y la potencia de cómputo suficiente, podrían seleccionarse
hashtags a voluntad para realizar una comparativa. Por ejemplo, podrían analizarse hashtags de apoyo a Donald Trump o Joe Biden para analizar el apoyo
de ambas carreras presidenciales en Twitter. Sin embargo, para este ejemplo se han buscado simplemente los 15 hashtags más utilizados.

## Arquitectura
Para lograrlo, se ha diseñado la siguiente arquitectura:

![](https://i.ibb.co/8sfFyGL/image.png)

Como se puede observar, un cliente descarga continuamente los tweets desde la API de Twitter. Este cliente se conecta a Spark mediante un socket TCP del sistema,
como se suele hacer habitualmente. Spark aplica un sencillo filtro para quedarse con los hashtags, y a continuación procesa el RDD haciendo un conteo de
estas palabras para finalmente devolver las 15 más utilizadas mediante una sencilla consulta SQL. Estos resultados son transmitidos a una web
mediante tecnología websocket, que permite una transferencia ligera y en tiempo real de los datos.
```
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

```

## Conclusión
Ha sido una práctica muy interesante para comprobar el enorme potencial de Spark, ya que con la infraestructura y recursos adecuados podrían llevarse
a cabo muy fácilmente tareas de análisis para recoger datos y conclusiones interesantes sobre prácticamente cualquier cosa. Con el auge del big data, 
Spark parece una tecnología de la que al menos deben conocerse sus conceptos básicos. Si bien es cierto que su curva de aprendizaje no resulta la más sencilla
del mundo, en general no es complicado una vez se entienden los conceptos básicos. Además, el uso de Python como
lenguaje ayuda a no complicar más su entendimiento.
