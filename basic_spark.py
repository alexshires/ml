#! /usr/local/spark-1.12.0/bin/spark-submit

import pyspark
from pyspark import SparkContext
#from pyspark import SparkContext._

if __name__ == '__main__':

    #set up spark
    sc = SparkContext('local', appName="TestApp" )
    textfile = sc.textFile("README.md")
    print textfile
    print textfile.count()
    print textfile.first()
    for i in range(10):
        print i 
    sc.stop()
