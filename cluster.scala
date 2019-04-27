//script to form clusters using spark mllib
//input to the script must only be numbers (it cannot handle strings)
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.StandardScaler

val data = sc.textFile("file:///D:/Bowling_Cluster_Final_Temp.csv")
val parsedData = data.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()

val scaler = new StandardScaler(withMean = true, withStd = true).fit(parsedData)
val parsedData1 = scaler.transform(parsedData)

val predict = sc.textFile("file:///D:/Bowling_Cluster_Final_Temp.csv")
val predictData = predict.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()
var min = 400000.0
var i_min = 0
var i=5
var WSSSE=0.0
val numIterations = 150
var numClusters = 1;
for(i <- 1 to 20)
{
numClusters = i
println(i)
val clusters = KMeans.train(parsedData1, numClusters, numIterations)

WSSSE = clusters.computeCost(parsedData1)
if(WSSSE<min)
{
	min = WSSSE
	i_min = i
}

}
println(s"Within Set Sum of Squared Errors = $WSSSE and i is = $i")
//numClusters = 5
val clusters1 = KMeans.train(parsedData1, i_min, numIterations)
clusters1.predict(parsedData1).foreach(println) 
