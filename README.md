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








# Basic RAG 

![IMG_0012](https://github.com/user-attachments/assets/0875eaae-2681-4d18-b90a-84a6fab212e7)

# Advanced RAG Techniques

## 1. Query Contruction

Naive RAG typically splits documents into chunks, embeds them, and retrieves chunks with high semantic similarity to a user question. But, this present a few problems: (1) document chunks may contain irrelevant content that degrades retrieval, (2) user questions may be poorly worded for retrieval, and (3) structured queries may need to be generated from the user question (e.g., for querying a vectorstore with metadata filtering or a SQL db).
Query transformation deals with transformations of the user's question before passing to the embedding model.
### i. Rewrite-Retrieve-Read - 
This paper uses an LLM to rewrite a user query, rather than using the raw user query to retrieve directly.
### ii. Step back prompting - 
This paper uses an LLM to generate a "step back" question. This can be use with or without retrieval. With retrieval, both the "step back" question and the original question are used to do retrieval, and then both results are used to ground the language model response.
 
The most basic and central place query transformation is used is in conversational chains to handle follow up questions. When dealing with follow up questions, there are essentially three options:
Just embed the follow up question. This means that if the follow up question builds on, or references the previous conversation, it will lose that question. For example, if I first ask "what can I do in Italy" and then ask "what type of food is there" - if I just embed "what type of food is there" I will have no context of where "there" is.
Embed the whole conversation (or last k messages). The problem with this is that if a follow up question is completely unrelated to previous conversation, then it may return completely irrelevant results that would distract during generation.
Use an LLM to do a query transformation!
### iii. Multi Query Retrieval - 
In this strategy, an LLM is used to generate multiple search queries. These search queries can then be executed in parallel, and the retrieved results passed in altogether. This is really useful when a single question may rely on multiple sub questions.
For example consider the following question:

Who won a championship more recently, the Red Sox or the Patriots?
This really requires two sub-questions:

"When was the last time the Red Sox won a championship?"
"When was the last time the Patriots won a championship?"

### iv. RAG-Fusion
A recent article builds off the idea of Multi-Query Retrieval. However, rather than passing in all the documents, they use reciprocal rank fusion to reorder the documents.

<img width="764" alt="Screen Shot 2025-01-09 at 12 12 52 AM" src="https://github.com/user-attachments/assets/f4471813-b9cb-4a9a-8c0d-b21e9bc8abc4" />


## 2. Query Routing
Query Routing is about giving our RAG app the power of decision-making. Query Routing is a technique that takes the query from the user and uses it to make a decision on the next action to take, from a list of predefined choices.

Query Routing is a module in our Advanced RAG architecture. It is usually found after any query rewriting or guardrails.

Which are the choices for the Query Router?
We have to define the choices that the Query Router can take beforehand. We must first implement each of the different strategies, and accompany each one with a nice description. It is very important that the description explains in detail what each strategy does, since this description will be what our router will base its decision on.

The choices a Query Router takes can be the following:

Retrieval from different data sources
We can catalog multiple data sources that contain information on different topics. We might have a data source that contains information about a product that the user has questions about. And another data source with information about our return policies, etc. Instead of looking for the answers for the user’s questions in all data sources, the query router can decide which data source to use based on the user query and the data source description.

Retrieval from different indexes
Query Routers can also choose to use a different index for the same data source.

For example, we could have an index for keyword based search and another for semantic search using vector embeddings. The Query Router can decide which of the two is best for getting the relevant context for answering the question, or maybe use both of them at the same time and combine the contexts from both.

We could also have different indexes for different retrieval strategies. For example, we could have a retrieval strategy based on summaries, or a sentence window retrieval strategy, or a parent-child retrieval strategy. The Query Router can analyze the specificity of the question and decide which strategy is best to use to get the best context.

Types of Query Routers

## i. LLM Selector Router
This solution gives a prompt to an LLM. The LLM completes the prompt with the solution, which is the selection of the right choice. The prompt includes all the different choices, each with its description, as well as the input query to base its decision on. The response of this query will be used to programmatically decide which path to take.

## ii. LLM Function Calling Router
This solution leverages the function calling capabilities (or tool using capabilities) of LLMs. Some LLMs have been trained to be able to decide to use some tools to get to an answer if they are provided for them in the prompt. Using this capability, each of the different choices is phrased like a tool in the prompt, prompting the LLM to choose which one of the tools provided is best to solve the problem of retrieving the right context for answering the query.

## iii. Semantic Router
This solution uses similarity search on the vector embedding representation of the user query. For each choice, we will have to write a few examples of a query that would be routed to this path. When a user query arrives, an embeddings model converts it to a vector representation and it is compared to the example queries for each router choice. The example with the nearest vector representation to the user query is chosen as the path the router must route to.
## iv. Zero-shot classification Router
For this type of router, a small LLM is selected to act as a router. This LLM will be finetuned using a dataset of examples of user queries and the correct routing for each of them. The finetuned LLM’s sole purpose will become to classify user queries. Small LLMs are more cost-effective and more than good enough for a simple classification task.

## v. Keyword router
Sometimes the use case is extremely simple. In this case, the solution could be to route one way or another depending on if some keywords are present in the user query. For example, if the query contains the word “return” we could use a data source with information useful about how to return a product. For this solution, a simple code implementation is enough, and therefore, no expensive model is needed.

Single choice routing vs Multiple choice routing
Depending on the use case, it will make sense for the router to just choose one path and run it. However, in some cases it also can make sense to use more than one choice for answering the same query. To answer a question that spans many topics, the application needs to retrieve information from many data sources. Or the response might be different based on each data source. Then, we can use all of them to answer the question and consolidate them in a single final answer.








