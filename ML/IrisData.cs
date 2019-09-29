using Microsoft.ML.Runtime.Api;

namespace ML
{
    // IrisData is used to provide training data, and as
    // input for prediction operations
    // - First 4 properties are inputs/features used to predict the label
    // - Label is what you are predicting, and is only set when training
    public class IrisData
    {
        [Column("4")]
        [ColumnName("Label")]
        public string Label;

        [Column("2")]
        public float PetalLength;

        [Column("3")]
        public float PetalWidth;

        [Column("0")]
        public float SepalLength;

        [Column("1")]
        public float SepalWidth;
    }

    // IrisPrediction is the result returned from prediction operations
    public class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels;
    }
}