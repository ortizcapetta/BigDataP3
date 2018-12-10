# coding=utf-8
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
model1res = spark.read.csv('Results/model1res.csv')
model2res = spark.read.csv('Results/model2res.csv')

model1res.createOrReplaceTempView("model1")

model1 = spark.sql("select _c1 as sentiment, count(*) "
                        "from model1 "
                        "group by sentiment "
                        "order by _c1 asc ")
model1.show()


model2res.createOrReplaceTempView("model2")
model2 = spark.sql("select _c1 as sentiment, count(*) "
                        "from model2 "
                        "group by sentiment "
                        "order by _c1 asc ")
model2.show()
