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
<img width="831" alt="Screen Shot 2025-01-15 at 4 26 11 PM" src="https://github.com/user-attachments/assets/8cf40467-b40f-4e0f-8d0e-8ed7afb91481" />


## 1. Query Transformation

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
<img width="897" alt="Screen Shot 2025-01-15 at 4 22 23 PM" src="https://github.com/user-attachments/assets/2a3711d7-47bb-43fb-ae08-c60208c05849" />

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

## 3. Query Construction








# Parameter Efficient Fine-Tuning LLMs (Large Language Models)

## LoRA (Low Rank Adaptation)
LoRA is a training method designed to expedite the training process of large language models, all while reducing memory consumption. By introducing pairs of rank-decomposition weight matrices, known as update matrices, to the existing weights, LoRA focuses solely on training these new added weights. This approach offers several advantages:

Preservation of pretrained Weights: LoRA maintains the frozen state of previously trained weights, minimizing the risk of catastrophic forgetting. This ensures that the model retains its existing knowledge while adapting to new data.
Portability of trained weights: The rank-decomposition matrices used in LoRA have significantly fewer parameters compared to the original model. This characteristic allows the trained LoRA weights to be easily transferred and utilized in other contexts, making them highly portable.
Integration with Attention Layers: LoRA matrices are typically incorporated into the attention layers of the original model. Additionally, the adaptation scale parameter allows control over the extent to which the model adjusts to new training data.
Memory efficiency: LoRA's improved memory efficiency opens up the possibily of running fine-tune tasks on less than 3x the required compute for a native fine-tune.

LoRA hyperparameters
### LoRA Rank
This determines the number of rank decomposition matrices. Rank decomposition is applied to weight matrices in order to reduce memory consumption and computational requirements. The original LoRA paper recommends a rank of 8 (r = 8) as the minimum amount. Keep in mind that higher ranks lead to better results and higher compute requirements. The more complex your dataset, the higher your rank will need to be.

To match a full fine-tune, you can set the rank to equal to the model's hidden size. This is, however, not recommended because it's a massive waste of resources. You can find out the model's hidden size by reading through the config.json or by loading the model with Transformers's AutoModel and using the model.config.hidden_size function:


from transformers import AutoModelForCausalLM
model_name = "huggyllama/llama-7b"      # can also be a local directory
model = AutoModelForCausalLM.from_pretrained(model_name)
hidden_size = model.config.hidden_size
print(hidden_size)

### LoRA Alpha

This is the scaling factor for the LoRA, which determines the extent to which the model is adapted towards new training data. The alpha value adjusts the contribution of the update matrices during the train process. Lower values give more weight to the original data and maintain the model's existing knowledge to a greater extent than higher values.

### LoRA Target Modules
Here you can determine which specific weights and matrices are to be trained. The most basic ones to train are the Query Vectors (e.g. q_proj) and Value Vectors (e.g. v_proj) projection matrices. The names of these matrices will differ from model to model. You can find out the exact names by running the following script:

from transformers import AutoModelForCausalLM
model_name = "huggyllama/llama-7b"      # can also be a local directory
model = AutoModelForCausalLM.from_pretrained(model_name)
layer_names = model.state_dict().keys()

for name in layer_names:
    print(name)


## QLoRA

QLoRA (Quantized Low Rank Adapters) is an efficient finetuning approach that reduces memory usage while maintaining high performance for large language models. It enables the finetuning of a 65B parameter model on a single 48GB GPU, while preserving full 16-bit fine-tuning task performance.

The key innovations of QLoRA include:

Backpropagation of gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters (LoRA).
Use of a new data type called 4-bit NormalFloat (NF4), which optimally handles normally distributed weights.
Double quantization to reduce the average memory footprint by quantizing the quantization constants.
Paged optimizers to effectively manage memory spikes during the finetuning process.



## Quantization

bitsandbytes is the easiest option for quantizing a model to 8 and 4-bit. 8-bit quantization multiplies outliers in fp16 with non-outliers in int8, converts the non-outlier values back to fp16, and then adds them together to return the weights in fp16. This reduces the degradative effect outlier values have on a model’s performance. 4-bit quantization compresses a model even further

### Compute data type
To speedup computation, you can change the data type from float32 (the default value) to bf16 using the bnb_4bit_compute_dtype parameter in BitsAndBytesConfig:

### Quant Type
NF4 is a 4-bit data type from the QLoRA paper, adapted for weights initialized from a normal distribution. You should use NF4 for training 4-bit base models. This can be configured with the bnb_4bit_quant_type parameter in the BitsAndBytesConfig:

### bnb_4bit_use_double_quant
Nested quantization
Nested quantization is a technique that can save additional memory at no additional performance cost. This feature performs a second quantization of the already quantized weights to save an additional 0.4 bits/parameter.

