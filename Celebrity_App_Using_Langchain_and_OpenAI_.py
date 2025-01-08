import os
from langchain.llms import OpenAI
import streamlit as st
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

from langchain.memory import ConversationBufferMemory

load_dotenv()


first_input_prompt = PromptTemplate(
    input_variables = ['name'],
    template = "Tell me about celebrity {name}"
)

second_input_prompt = PromptTemplate(
    input_variables = ['person'],
    template="when was {person} born"
)

third_input_prompt = PromptTemplate(
    input_variables = ['dob'],
    templeate = "Mention 5 major events happended around {dob} in the world"
)

st.title('Langchain demo with openai API')

input_text =  st.text_input('Search the topic u want')

llm = OpenAI(temperature = 0.8)

chain = LLMChain(llm = llm, prompt = first_input_prompt, verbose = True, output_key = 'title')

chain2 = LLMChain(llm = llm, prompt = second_input_prompt,verbase = True, output_key = 'dob')

chain3 = LLMChain(llm = llm, prompt = third_input_prompt, verbose = True, output_key = 'description')
parent_chain = SimpleSequentialChain(chains = [chain, chain2], verbose=True)



if input_text:
    st.write(parent_chain.run(input_text))


