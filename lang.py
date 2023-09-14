
import os
import openai
from constant import openai_key,hugging_face,google_searchAPI_key

import streamlit as sl

from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain,SimpleSequentialChain,SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentType,initialize_agent,load_tools
from langchain.document_loaders import PyPDFLoader

os.environ['OPENAI_API_KEY'] = openai_key
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hugging_face
os.environ['SERPAPI_API_KEY'] = google_searchAPI_key

llm_model = "gpt-3.5-turbo"

llm = OpenAI(temperature=0.0)
# tools=load_tools(["serpapi","llm-math"],llm=llm)
tools=load_tools(['wikipedia','llm-math'],llm=llm)

agent=initialize_agent(tools=tools,llm=llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)

# agent.run("How was run virat kohli scored in previous match?")
# agent.run("In which year Titanic movie was released and why it is famous?")

prompt1 = PromptTemplate(
    input_variables=['Lunch'],
    template="I want to open a restaurent for {Lunch} for suggest me fancy name"
)

chain1 = LLMChain(llm=llm,prompt=prompt1,output_key='name')

prompt2 =PromptTemplate(
    input_variables=['name'],
    template="Suggest some Menu items for {name}"
)

chain2=LLMChain(llm=llm,prompt=prompt2,output_key='menu_item')

parentchain = SequentialChain(
    chains=[chain1,chain2],
    input_variables=['Lunch'],
    output_variables=['name','menu_item']
)

# print(parentchain({"Lunch":"India"}))

memory1=ConversationBufferMemory()
chains = LLMChain(
    llm=llm,
    prompt=prompt1,
    memory=memory1
)

# print(chains({"Lunch":"America"}))
# print(chains({"Lunch":"India"}))
#
# print(chains.memory.buffer)


conv = ConversationChain(llm=llm)
# print(conv.prompt.template)

# print(conv.run("Who won the World cup in 1975?"))
# print(conv.run("Who was the Captain during that time?"))
#
# print(conv.memory.buffer)

memory2=ConversationBufferWindowMemory(k=2)

convo = ConversationChain(
    llm=llm,
    memory=memory2
)

# print(convo.run("In which year India won the World cup?"))
# print(convo.run("Who was the captain of the team that time?"))
# print(convo.run("Give me who scored more runs?"))
#
# print(convo.memory.buffer)

# loader = PyPDFLoader("Fine-Tune Your Own Llama 2 Model in a Colab Notebook _ Towards Data Science.pdf")
# pages = loader.load()
#
# print(pages)








