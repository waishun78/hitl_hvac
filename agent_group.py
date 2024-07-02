import pandas as pd
from constants import *
from agent import Agent

# Add noise to the PMV score to incorporate for anomalous agents
class AgentGroup():
    def __init__(self, group_size=-1, agents=None, font=None, screen=None, model=None):
        self.agents = [] if agents is None else agents
        self.group_size = group_size if group_size >= 0 else len(agents)
        self.font = font
        self.screen = screen
        self.nextAgentId = 0 
        self.data = None
        self.group_data_df = None
        self.model = model

    def read_data(self):
        data_df = pd.read_csv(DATA_FILE, low_memory=False)
        if self.group_size > len(data_df):
            raise ValueError("Group size is larger than the size of the data")
        self.data = data_df.sample(n=self.group_size).to_dict("records")

    def generate(self):
        self.read_data()
        for d in self.data:
            self.agents.append(Agent(id=self.nextAgentId, color=AGENT_OUT_COLOR, data=d, font=self.font, screen=self.screen).randomOut())
            self.nextAgentId += 1
        return self.agents

    def get_group_data_df(self):
        if self.group_data_df is not None:
            return self.group_data_df
        
        data_rows = []
        for agent in self.agents:
            data_rows.append(agent.data)
        self.group_data_df = pd.DataFrame(data_rows)
        return self.group_data_df

    def predict(self, temp, ambient_temp, alpha=1, beta=0.03):
        q = -m * c_p * abs(temp - ambient_temp)
        power = q / (TIME_INTERVAL / timedelta(hours=1)) / 1000

        if len(self.agents) > 0:
            df = self.get_group_data_df()
            pred = self.model.predict_score(df['user'].values, temp)
            # print('Predicted Comfort: ', pred)
            # print('Power Consumed: ', power)
            # print('Formula for Sum: ', alpha * VOLUME * pred.sum() / (len(df) ** 0.4) + beta * power)
            # print('Formula for Mean: ', alpha * VOLUME * pred.mean() / (len(df) ** 0.4) + beta * power)
            # print('='*20)
            return (alpha * VOLUME * pred.sum() / (len(df) ** 0.4) + beta * power), (alpha * VOLUME * pred.mean() / (len(df) ** 0.4) + beta * power)
        else:
            print('No agents in group')
            return power
