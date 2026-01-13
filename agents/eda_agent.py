import pandas as pd
import numpy as np

def detect_outliers(series):
    Q1=series.quantile(0.25)
    Q3=series.quantile(0.75)
    IQR=Q3-Q1
    lower=Q1-1.5*IQR
    upper=Q3+1.5*IQR

    return int(((series<lower) | (series>upper)).sum())

def eda_agent(state):
    print("EDA AGENT HAS STARTED......")


    df=state['df']
    target=state['target']
    task_type=state['task_type']

    shape=df.shape
    dtype=df.dtypes.astype(str).to_dict()

    missing=df.isna().sum().to_dict()

    nums_cols=df.select_dtypes(include=np.number).columns.tolist()
    cats_cols=df.select_dtypes(exclude=np.number).columns.tolist()

    numerical_summary=df[nums_cols].describe().to_dict() if nums_cols else {}

    categorical_summary={
        col:df[col].value_counts().head(5).to_dict()
        for col in cats_cols
    }

    corelations={}
    if target in nums_cols:
        corr_series=df[nums_cols].corr()[target].sort_values(ascending=False)
        corelations=corr_series.to_dict()

    outliers={
        col:detect_outliers(df[col])
        for col in nums_cols
    }

    class_balance={}
    if task_type=='classification':
        #normalize changes count into probabilities
        class_balance=df[target].value_counts(normalize=True).to_dict()


    state['eda_summary']={
        'shape':shape,
        'dtype':dtype,
        'missing_values':missing,
        'nums_cols':nums_cols,
        'cats_cols':cats_cols,
        'numerical_summary':numerical_summary,
        'categorical_summary':categorical_summary,
        'corelations':corelations,
        'outliers':outliers,
        'class_balance':class_balance
    }

    print("EDA COMPLETED.....")
    return state
