import pandas as pd

TARGET_VALUES=['target','outcome','y','class','label']

def detect_target_column(df):


    #This is for keyword detection i.e. if column name matches the keyword
    for col in df.columns:
        if col.lower() in TARGET_VALUES:
            return col
        

    #lowest unique count i.e. we try to find which column has lowest unique values
    # unique_counts=df.nunique()
    # candidate=unique_counts.idxmin()

    # if unique_counts[candidate]>1 and unique_counts[candidate] < len(df)*0.2:
    #     return candidate
    
    #last resort go for the last column
    return df.columns[-1];


def detect_task_type(df,target_col):
    target=df[target_col]

    if target.dtype =='object' or target.dtype.name == 'category':
        return 'classification'
    
    unique_vals=target.nunique()

    if unique_vals <=20:
        return 'classification'
    
    return 'regression'

def orchestrator_agent(state):
    print("Orchestration agent has started")

    df=pd.read_csv(state['csv_path'])

    target_col=detect_target_column(df)
    task_type=detect_task_type(df,target_col)

    feature_cols=[col for col in df.columns if col!=target_col]


    #since state is a dictionary and update is built-in function of dictionary
    state.update({
        "df":df,
        "target":target_col,
        "task_type":task_type,
        "feature_cols":feature_cols
    })


    print(f"target detected:{target_col}")
    print(f"task type is {task_type}")

    return state



