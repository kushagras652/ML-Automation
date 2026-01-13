# import json
# from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage,HumanMessage
import json


model=ChatOpenAI(
    model='gpt-4.1-mini',
    temperature=0.3
)

def build_prompt(state):
        return f"""
You are a senior data scientist.

Given the following analysis results, generate clear and concise insights
for a non-technical stakeholder.

Dataset Task Type: {state["task_type"]}
Target Variable: {state["target"]}

EDA Summary:
{json.dumps(state["eda_summary"], indent=2)}

Model Performance:
{json.dumps(state["model_results"], indent=2)}

Best Model:
{state["best_model"]}

Your response should include:
1. Overall data quality assessment
2. Key patterns or issues (missing values, imbalance, outliers)
3. Why the best model performed well
4. Important features (infer from correlations)
5. Risks or limitations
6. Actionable recommendations

Avoid technical jargon.
"""

def insight_agent(state):
        print("INSIGHT AGENT STARTED....")

        prompt=build_prompt(state)

        messages=[
                SystemMessage(content="You generate analytical insights"),
                HumanMessage(content=prompt)
        ]

        response=model.invoke(messages)

        state['final_insights']=response.content

        print("Insight generated...")

        return state