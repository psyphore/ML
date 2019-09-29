using Microsoft.ML.Data;
using System;

namespace Models.Billing
{
    public class PredictedInsurerBill
    {
        [ColumnName("Score")]
        public float SPV;
    }
}
