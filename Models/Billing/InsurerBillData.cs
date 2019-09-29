using Microsoft.ML.Data;

namespace Models.Billing
{
    public class InsurerBillData
    {
        [LoadColumn(0)]
        public string ChannelName;

        [LoadColumn(2), VectorType()]
        public string DateSold;

        [LoadColumn(1)]
        public float ProductId;

        [LoadColumn(3), ColumnName("Label")]
        public float SPV;
    }
}
