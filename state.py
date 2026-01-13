from typing import TypedDict,Dict,Any

class AgentState(TypedDict):
    csv_path:str
    df:Any
    target:str
    task_type:str
    feature_cols:list
    eda_summary:Dict
    model_results:Dict
    best_model:str
    final_insights:str