import scala.util.Random
import breeze.linalg._

import java.io.{File, PrintWriter}
import scala.io._
import org.apache.spark.SparkContext._

import scala.io._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd._
import org.apache.log4j.Logger
import org.apache.log4j.Level

import scala.collection._
import org.apache.spark.sql.{Encoders, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.{Pipeline, classification}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}




case class Person(
                   id: Int,
                   gender: String,
                   age: Int,
                   occupation: String,
                   sleepDuration: Double,
                   sleepQuality: Double,
                   physicalActivity: Double,
                   stressLevel: Double,
                   bmiCategory: String,
                   bloodPressure: String,
                   heartRate: Double,
                   dailySteps: Double,
                   sleepDisorder: String
                 )

object FinalProject {
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  val spark: SparkSession = SparkSession.builder()
    .appName("CSVReader")
    .master("local[1]")
    .getOrCreate()


  def main(args: Array[String]): Unit = {
    // Read in data
    val schema = StructType(Seq(
      StructField("id", IntegerType, nullable = true),
      StructField("gender", StringType, nullable = true),
      StructField("age", IntegerType, nullable = true),
      StructField("occupation", StringType, nullable = true),
      StructField("sleepDuration", DoubleType, nullable = true),
      StructField("sleepQuality", DoubleType, nullable = true),
      StructField("physicalActivity", DoubleType, nullable = true),
      StructField("stressLevel", DoubleType, nullable = true),
      StructField("bmiCategory", StringType, nullable = true),
      StructField("bloodPressure", StringType, nullable = true),
      StructField("heartRate", DoubleType, nullable = true),
      StructField("dailySteps", DoubleType, nullable = true),
      StructField("sleepDisorder", StringType, nullable = true)
    ))
    val file = "src/main/scala/Sleep_health_and_lifestyle_dataset.csv"
    val df_sleep = spark.read.schema(schema).csv(file).as[Person](Encoders.product[Person])

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = df_sleep.randomSplit(Array(0.8, 0.2))

    val disorderToNum = new StringIndexer()
      .setInputCol("sleepDisorder")
      .setOutputCol("SD_quant")


    // Setup Model 1
    val features1 = Array("sleepDuration", "sleepQuality")
    val assembler1 = new VectorAssembler()
      .setInputCols(features1)
      .setOutputCol("features1")
    val lr1 = new LogisticRegression()
      .setLabelCol("SD_quant")
      .setFeaturesCol("features1")
    val pipeline1 = new Pipeline()
      .setStages(Array(assembler1, disorderToNum, lr1))

    // Setup Model 2
    val features2 = Array("physicalActivity", "heartRate", "dailySteps")
    val assembler2 = new VectorAssembler()
      .setInputCols(features2)
      .setOutputCol("features2")
    val lr2 = new LogisticRegression()
      .setLabelCol("SD_quant")
      .setFeaturesCol("features2")
    val pipeline2 = new Pipeline()
      .setStages(Array(assembler2, disorderToNum, lr2))

    // Setup Model 3
    val occupationIndexer = new StringIndexer()
      .setInputCol("occupation")
      .setOutputCol("occupationIndex")
      .fit(df_sleep)
    val occupationEncoder = new OneHotEncoder()
      .setInputCol("occupationIndex")
      .setOutputCol("occupationVec")
    val features3 = Array("stressLevel", "occupationVec")
    val assembler3 = new VectorAssembler()
      .setInputCols(features3)
      .setOutputCol("features3")
    val lr3 = new LogisticRegression()
      .setLabelCol("SD_quant")
      .setFeaturesCol("features3")
    val pipeline3 = new Pipeline()
      .setStages(Array(occupationIndexer, occupationEncoder, assembler3, disorderToNum, lr3))

    // Define hyper-parameter grids for each model
    val paramGrid1 = new ParamGridBuilder()
      .addGrid(lr1.regParam, Array(0.0, 0.1, 1.0)) // Regularization parameter
      .build()

    val paramGrid2 = new ParamGridBuilder()
      .addGrid(lr2.regParam, Array(0.0, 0.1, 1.0)) // Regularization parameter
      .build()

    val paramGrid3 = new ParamGridBuilder()
      .addGrid(lr3.regParam, Array(0.0, 0.1, 1.0)) // Regularization parameter
      .build()

    // Define evaluator
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("SD_quant")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    // Define CrossValidator for each model
    val cv1 = new CrossValidator()
      .setEstimator(pipeline1)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid1)
      .setNumFolds(5) // Use 5-fold cross-validation

    val cv2 = new CrossValidator()
      .setEstimator(pipeline2)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid2)
      .setNumFolds(5) // Use 5-fold cross-validation

    val cv3 = new CrossValidator()
      .setEstimator(pipeline3)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid3)
      .setNumFolds(5) // Use 5-fold cross-validation

    // Perform hyper-parameter tuning and fit models
    val cvModel1 = cv1.fit(trainingData)
    val cvModel2 = cv2.fit(trainingData)
    val cvModel3 = cv3.fit(trainingData)

    // Predict sleep disorder for test data using best models
    val predictions1 = cvModel1.transform(testData)
    val predictions2 = cvModel2.transform(testData)
    val predictions3 = cvModel3.transform(testData)

    // Evaluate Model performance
    val accuracy1 = evaluator.evaluate(predictions1)
    val accuracy2 = evaluator.evaluate(predictions2)
    val accuracy3 = evaluator.evaluate(predictions3)

    println(s"Model 1 Accuracy: $accuracy1")
    println(s"Model 2 Accuracy: $accuracy2")
    println(s"Model 3 Accuracy: $accuracy3")

    spark.stop()
  }
}