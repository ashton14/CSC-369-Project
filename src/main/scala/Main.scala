import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import breeze.linalg._
import scala.util.Random

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

object Main {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.appName("Sleep Health and Lifestyle").getOrCreate()
    import spark.implicits._

    val file = "Sleep_health_and_lifestyle_dataset.csv"
    val lines = spark.read.textFile(file).rdd
    val header = lines.first()
    val data = lines.filter(_ != header).map(parseLine)

    val numFolds = 5
    val ks = List(3, 5, 10)

    for (k <- ks) {
      println(s"K = $k")
      // Model 1: Using age, sleep duration, and sleep quality
      val accuracyModel1 = crossValidate(data, k, numFolds, distanceModel1, spark)
      val roundedAccuracy1 = BigDecimal(accuracyModel1).setScale(2, BigDecimal.RoundingMode.HALF_UP)
      println(f"Model 1 Average Accuracy: $roundedAccuracy1%.2f%%")

      // Model 2: Using age, physical activity, heart rate, and daily steps
      val accuracyModel2 = crossValidate(data, k, numFolds, distanceModel2, spark)
      val roundedAccuracy2 = BigDecimal(accuracyModel2).setScale(2, BigDecimal.RoundingMode.HALF_UP)
      println(f"Model 2 Average Accuracy: $roundedAccuracy2%.2f%%")

      // Model 3: Using age and stress level
      val accuracyModel3 = crossValidate(data, k, numFolds, distanceModel3, spark)
      val roundedAccuracy3 = BigDecimal(accuracyModel3).setScale(2, BigDecimal.RoundingMode.HALF_UP)
      println(f"Model 3 Average Accuracy: $roundedAccuracy3%.2f%%")
    }

    spark.stop()
  }

  def parseLine(line: String): Person = {
    val cols = line.split(",")
    Person(
      cols(0).toInt,
      cols(1),
      cols(2).toInt,
      cols(3),
      cols(4).toDouble,
      cols(5).toDouble,
      cols(6).toDouble,
      cols(7).toDouble,
      cols(8),
      cols(9),
      cols(10).toDouble,
      cols(11).toDouble,
      cols(12)
    )
  }

  def knn(data: RDD[Person], target: Person, k: Int, distanceFunc: (Person, Person) => Double): List[Person] = {
    data
      .filter(_.id != target.id) // Exclude the target itself
      .map(person => (person, distanceFunc(person, target)))
      .sortBy(_._2)
      .take(k)
      .map(_._1)
      .toList
  }

  def distanceModel1(p1: Person, p2: Person): Double = {
    val vec1 = DenseVector(p1.age, p1.sleepDuration, p1.sleepQuality)
    val vec2 = DenseVector(p2.age, p2.sleepDuration, p2.sleepQuality)
    norm(vec1 - vec2)
  }

  def distanceModel2(p1: Person, p2: Person): Double = {
    val vec1 = DenseVector(p1.age, p1.physicalActivity, p1.heartRate, p1.dailySteps)
    val vec2 = DenseVector(p2.age, p2.physicalActivity, p2.heartRate, p2.dailySteps)
    norm(vec1 - vec2)
  }

  def distanceModel3(p1: Person, p2: Person): Double = {
    val vec1 = DenseVector(p1.age, p1.stressLevel)
    val vec2 = DenseVector(p2.age, p2.stressLevel)
    norm(vec1 - vec2)
  }

  def predict(neighbors: List[Person]): String = {
    neighbors.groupBy(_.sleepDisorder).maxBy(_._2.size)._1
  }

  def evaluateModel(trainingData: RDD[Person], testData: RDD[Person], k: Int, distanceFunc: (Person, Person) => Double): Double = {
    val correctPredictions = testData.map { testPerson =>
      val neighbors = knn(trainingData, testPerson, k, distanceFunc)
      val predictedDisorder = predict(neighbors)
      if (predictedDisorder == testPerson.sleepDisorder) 1 else 0
    }.sum()
    (correctPredictions / testData.count()) * 100
  }

  def crossValidate(data: RDD[Person], k: Int, numFolds: Int, distanceFunc: (Person, Person) => Double, spark: SparkSession): Double = {
    val foldSize = data.count() / numFolds
    val shuffledData = data.map((_, Random.nextDouble())).sortBy(_._2).map(_._1) // Shuffle data
    val folds = shuffledData.zipWithIndex().groupBy(_._2 / foldSize).mapValues(_.map(_._1).collect().toList).collect().toList.map(_._2)

    val accuracies = (0 until numFolds).map { i =>
      val testFold = folds(i)
      val trainingFolds = folds.take(i) ++ folds.drop(i + 1)
      val trainingData = spark.sparkContext.parallelize(trainingFolds.flatten)

      evaluateModel(trainingData, spark.sparkContext.parallelize(testFold), k, distanceFunc)
    }

    accuracies.sum / numFolds
  }
}
