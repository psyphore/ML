using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Models.Billing;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Services.Billing
{
    public class InsurerBillingService
    {
        private const string testDataFilePath = "regressionInsurerBillingData.csv";
        private const string trainingDataFilePath = "trainingInsurerBillingData.csv";
        private readonly MLContext mlContext;

        public InsurerBillingService()
        {
            mlContext = new MLContext();
        }

        public void Trainer()
        {
            var data = mlContext.Data.LoadFromTextFile<InsurerBillData>(trainingDataFilePath, separatorChar: ',', hasHeader: true);
            var dataSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            IDataView trainData = dataSplit.TrainSet;

            IDataView testData = dataSplit.TestSet;

            // Define Data Prep Estimator
            // 1. Concatenate Size and Historical into a single feature vector output to a new column called Features
            // 2. Normalize Features vector
            IEstimator<ITransformer> dataPrepEstimator = mlContext.Transforms
                .Concatenate("ChannelName", "ProductId", "DateSold", "SPV")
                .Append(mlContext.Transforms.NormalizeMinMax("SPV"));

            // Create data prep transformer
            ITransformer dataPrepTransformer = dataPrepEstimator.Fit(trainData);

            // Apply transforms to training data
            IDataView transformedTrainingData = dataPrepTransformer.Transform(trainData);

            // train model
            // Define StochasticDualCoordinateAscent regression algorithm estimator
            var sdcaEstimator = mlContext.Regression.Trainers.Sdca();

            // Build machine learning model
            var trainedModel = sdcaEstimator.Fit(transformedTrainingData);

            var trainedModelParameters = trainedModel.Model as LinearRegressionModelParameters;

            // Evaluate model quality
            // Measure trained model performance
            // Apply data prep transformer to test data
            IDataView transformedTestData = dataPrepTransformer.Transform(testData);

            // Use trained model to make inferences on test data
            IDataView testDataPredictions = trainedModel.Transform(transformedTestData);

            // Extract model metrics and get RSquared
            RegressionMetrics trainedModelMetrics = mlContext.Regression.Evaluate(testDataPredictions);
            double rSquared = trainedModelMetrics.RSquared;

            // Save Trained Model
            mlContext.Model.Save(trainedModel, data.Schema, "model.zip");
        }

        public void LoadModel()
        {
            //Define DataViewSchema for data preparation pipeline and trained model
            DataViewSchema modelSchema;

            // Load trained model
            ITransformer trainedModel = mlContext.Model.Load("model.zip", out modelSchema);
        }

        public float SinglePredict(InsurerBillData bill)
        {
            // Load Trained Model
            DataViewSchema predictionPipelineSchema;
            ITransformer predictionPipeline = mlContext.Model.Load("model.zip", out predictionPipelineSchema);

            // Create PredictionEngines
            PredictionEngine<InsurerBillData, PredictedInsurerBill> predictionEngine = mlContext.Model.CreatePredictionEngine<InsurerBillData, PredictedInsurerBill>(predictionPipeline);


            // Input Data
            //var inputData = new InsurerBillData
            //{
            //    ChannelName = "Bidvest",
            //    DateSold = DateTime.Now.ToShortDateString(),
            //    ProductId = 15,
            //    SPV = 0f
            //};

            // Get Prediction
            PredictedInsurerBill prediction = predictionEngine.Predict(bill);

            // Get Predictions
            var value = prediction.SPV;

            return value;
        }

        public IEnumerable<float> MultiplePredictions(IEnumerable<InsurerBillData> bills)
        {
            // Load Trained Model
            DataViewSchema predictionPipelineSchema;
            var predictionPipeline = mlContext.Model.Load("model.zip", out predictionPipelineSchema);

            // Create PredictionEngines
            var predictionEngine = mlContext.Model.CreatePredictionEngine<InsurerBillData, PredictedInsurerBill>(predictionPipeline, predictionPipelineSchema);

            IDataView newData = mlContext.Data.LoadFromEnumerable<InsurerBillData>(bills);

            // Get Prediction
            // Predicted Data
            IDataView predictions = predictionPipeline.Transform(newData);

            // Get Predictions
            float[] scoreColumn = predictions.GetColumn<float>("SPV").ToArray();

            return scoreColumn;
        }
    }
}