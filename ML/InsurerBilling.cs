using Microsoft.ML.Legacy;
using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Legacy.Models;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Legacy.Transforms;
using System;

namespace ML
{
    public class InsurerBilling
    {
        private const string testDataFilePath = "regressionInsurerBillingData.csv";
        private const string trainingDataFilePath = "trainingInsurerBillingData.csv";

        public static void Trainer()
        {
            var pipeline = new LearningPipeline
            {
                new TextLoader(trainingDataFilePath).CreateFrom<InsurerBillData>(useHeader: true, separator: ','),
                //new ColumnCopier(("Score", "Label")),
                new CategoricalOneHotVectorizer("ChannelName", "ProductId"),
                new ColumnConcatenator("Features", "ChannelName", "ProductId"),
                new FastTreeRegressor()
                //new GeneralizedAdditiveModelRegressor(),
            };

            var model = pipeline.Train<InsurerBillData, PredictedInsurerBill>();

            var testData = new TextLoader(testDataFilePath).CreateFrom<InsurerBillData>(useHeader: true, separator: ',');

            var evaluator = new RegressionEvaluator();
            var metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine($"RMS - {metrics.Rms}");
            Console.WriteLine($"R^2 - {metrics.RSquared}");

            var test = new InsurerBillData { ChannelName = "" };
            var prediction = model.Predict(test);
            Console.WriteLine($"Predicted SPV - {prediction.SPV}");

            Console.ReadLine();
        }
    }
}