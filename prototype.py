
import os
import openai
import pandas as pd
from constant import openai_key

import warnings
warnings.filterwarnings("ignore")

from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = openai_key

url = "intel.csv"

df = pd.read_csv(url)
# print(df.shape)
#
# print(df.head())

llm = OpenAI(temperature=0.7)

agent = create_pandas_dataframe_agent(llm,df,verbose=True)

# agent.run("How many rows are there?")

print(df.info())

df1 = df.copy()

df2=df1.copy()

df2['Lat_mul'] = df1['Lat']*2

agent1 = create_pandas_dataframe_agent(llm,[df,df1,df2],verbose=True)

agent1.run("Diifferenc in the count of columns in three dataframe")
