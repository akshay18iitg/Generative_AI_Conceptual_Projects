# Generative_AI_Conceptual_Projects

In this repo, I will try to create basic applications using GEN_AI.

## LangChain

LangChain is open source framework for creating LLM powered applications.
Some of building Blocks are:

PromptTemplate - These help to generate customized , generic prompt templates which can be used to query LLMs multiple times. It takes input_variables and template as input.

Chains - These help to organize LLMs, input prompt and their outputs and can help to perform complex prompting. There are various types of Chains. They are:
1. LLMChains: It takes one prompt imput and one LLM. Return one output.
2. Simple Sequential Chains : It can combine multiple LLMChains sequentially. The output is output of final chain.
3. Sequential Chian : It can combin multiple LLMChians sequentially. It can output all the intermediate LLMChains output.


Memory - This helps to maintain the context of conversation. 
