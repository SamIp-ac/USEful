from pyspark.sql import SparkSession

#
spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()
# Pivot Table
data = [("Banana", 1000, "USA"), ("Carrots", 1500, "USA"), ("Beans", 1600, "USA"),
        ("Orange", 2000, "USA"), ("Orange", 2000, "USA"), ("Banana", 400, "China"),
        ("Carrots", 1200, "China"), ("Beans", 1500, "China"), ("Orange", 4000, "China"),
        ("Banana", 2000, "Canada"), ("Carrots", 2000, "Canada"), ("Beans", 2000, "Mexico")]

columns = ["Product", "Amount", "Country"]
df = spark.createDataFrame(data=data, schema=columns)
df.printSchema()
df.show(truncate=False)

pivotDF = df.groupBy('Product').pivot('Country').sum('Amount')
pivotDF.printSchema()
pivotDF.show(truncate=False)

pivotDF = df.groupBy('Product', 'Country') \
      .sum('Amount') \
      .groupBy('Product') \
      .pivot('Country') \
      .sum('sum(Amount)')
pivotDF.printSchema()
pivotDF.show(truncate=False)

# Cross Table
df.crosstab('Amount', 'Product').show()
