/**
* This is re-implementation of http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html 
* 
*/


def ClassificationReport( predictions: RDD[(Double, Double)]) = {

val labelCounts = predictions.values.countByValue()
val mcm = new MulticlassMetrics(predictions)

val roundedAccuracy = (math rint predictions.map { case (prediction, label) =>
      if (label == prediction) 1 else 0
    }.sum().toDouble / predictions.count().toDouble *10000)/100

println(f"Percent Correct = $roundedAccuracy")
println()
println("Precision: The percent of times the class was predicted and the prediction was correct")
println("Recall: The percent of true labels that were identified by the model")
println("F1-score: The average of precision and recall")
println("Support: The number of true predictions that were predicted")
println()
println("Class\tPrecision\tRecall\tF1Score\tSupport")
mcm.labels.map{x =>
  val label = x.toInt
   val precision = (math rint mcm.precision(label) *10000)/10000
   val recall = (math rint mcm.recall(label)*10000)/10000
   val f1 = (math rint mcm.fMeasure(label)*10000)/10000

   (label, precision, recall, f1)
   }.foreach{case(label,precision,recall,f1) => println("%d\t%2.2f\t%2.2f\t%2.2f".format(label,precision,recall,f1))}
println()
println("Confusion Matrix:")
println( mcm.confusionMatrix)
}
