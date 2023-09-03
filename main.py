
import os
import openai
print(openai.__version__)

import langchain
import streamlit as st
from constant import openai_key

from langchain.llms import OpenAI

os.environ['OPENAI_API_KEY']= openai_key

print(langchain.__version__)
print(st.__version__)
# print(openai_key)

#streamlit framework

st.title("Langchain Demo with OpenAI API")
input_text = st.text_input("Enter the Topic you want")


#OPENAI API
llm = OpenAI(temperature=0.8)

if input_text:
    st.write(llm(input_text))
