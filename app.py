# from agents.orchestration import orchestrator_agent
# from agents.eda_agent import eda_agent
# from agents.ml_agent import ml_agent
# from agents.insight_agent import insight_agent

from graph import build_graph

app=build_graph()

initial_state={
    'csv_path':"data/house_price_regression_dataset.csv"
}

# state=orchestrator_agent(state)
# state=eda_agent(state)
# state=ml_agent(state)
# state=insight_agent(state)
# print(state['final_insights'])

final_state=app.invoke(initial_state)

print(final_state)