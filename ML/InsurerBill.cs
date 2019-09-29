using Microsoft.ML.Runtime.Api;
using System;

namespace ML
{
    public class InsurerBillData
    {
        [Column("0")]
        public string ChannelName;

        [Column("2")]
        public string DateSold;

        [Column("1")]
        public float ProductId;

        [Column("3", name: "Label")]
        public float SPV;
    }

    public class PredictedInsurerBill
    {
        [ColumnName("Score")]
        public float SPV;
    }
}