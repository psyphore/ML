using Microsoft.ML.Legacy;
using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Legacy.Transforms;
using System;
using System.IO;
using TensorFlow;

namespace ML
{
    /// <summary>
    /// https://blogs.msdn.microsoft.com/dotnet/2018/09/12/announcing-ml-net-0-5/?utm_source=vs_developer_news&utm_medium=referral
    /// https://github.com/migueldeicaza/TensorFlowSharp
    /// https://github.com/tensorflow/models
    ///
    /// https://github.com/dotnet/machinelearning-samples
    /// https://docs.microsoft.com/en-us/dotnet/machine-learning/
    /// https://www.microsoft.com/net/learn/machine-learning-and-ai/get-started-with-ml-dotnet-tutorial
    /// </summary>
    public class Service
    {
        public static void GraphTrainner()
        {
            using (var graph = new TFGraph())
            {
                graph.Import(File.ReadAllBytes("MySavedModel"));
                var session = new TFSession(graph);
                var runner = session.GetRunner();
                TFTensor tensor = null;
                runner.AddInput(graph["input"][0], tensor);
                runner.Fetch(graph["output"][0]);

                var output = runner.Run();

                // Fetch the results from output:
                TFTensor result = output[0];
            }
        }

        public static void IrisDataRunner()
        {
            // STEP 2: Create a pipeline and load your data
            var pipeline = new LearningPipeline();

            // If working in Visual Studio, make sure the 'Copy to Output Directory'
            // property of iris-data.txt is set to 'Copy always'
            string dataPath = "iris-data.txt";
            pipeline.Add(new TextLoader(dataPath).CreateFrom<IrisData>(separator: ','));

            // STEP 3: Transform your data
            // Assign numeric values to text in the "Label" column, because only
            // numbers can be processed during model training
            pipeline.Add(new Dictionarizer("Label"));

            // Puts all features into a vector
            pipeline.Add(new ColumnConcatenator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));

            // STEP 4: Add learner
            // Add a learning algorithm to the pipeline.
            // This is a classification scenario (What type of iris is this?)
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            // Convert the Label back into original text (after converting to number in step 3)
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            // STEP 5: Train your model based on the data set
            var model = pipeline.Train<IrisData, IrisPrediction>();

            // STEP 6: Use your model to make a prediction
            // You can change these numbers to test different predictions
            var prediction = model.Predict(new IrisData
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f,
            });

            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabels}");
        }

        public static void SessionTraining()
        {
            using (var session = new TFSession())
            {
                var graph = session.Graph;

                var a = graph.Const(2);
                var b = graph.Const(3);
                Console.WriteLine("a=2 b=3");

                // Add two constants
                var addingResults = session.GetRunner().Run(graph.Add(a, b));
                var addingResultValue = addingResults.GetValue();
                Console.WriteLine("a+b={0}", addingResultValue);

                // Multiply two constants
                var multiplyResults = session.GetRunner().Run(graph.Mul(a, b));
                var multiplyResultValue = multiplyResults.GetValue();
                Console.WriteLine("a*b={0}", multiplyResultValue);
            }
        }
    }
}