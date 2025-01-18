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

## 3. Query Construction [[link](https://blog.langchain.dev/query-construction/)]

Query Construction is useful when we have to construct a query specific to a database. This involves converting natural language query to database specific query. The below table mentiones which query is usetable for which database

<img width="658" alt="Screen Shot 2025-01-16 at 8 13 33 PM" src="https://github.com/user-attachments/assets/3064bd79-95a3-43fd-b47d-af6b54bc5da2" />

### i. Text to Metadata Filter

Vectorstores equipped with metadata filtering enable structured queries to filter embedded unstructured documents. The self-query retriever can translate natural language queries into these structured queries using a few steps:
Data Source Definition: At its core, the self-query retriever is anchored by a clear specification of the relevant metadata files (e.g., in the context of song retrieval, this might be artist, length, and genre).
User Query Interpretation: Given a natural language question, the self-query retriever will isolate the query (for semantic retrieval) and the filter for metadata filtering. For instance, a query for songs by Taylor Swift or Katy Perry about teenage romance under 3 minutes long in the dance pop genre is decomposed into a filter and query.
Logical Condition Extraction: The filter itself is crafted from vectorstore defined comparators and operators like eq for equals or lt for less than.
Structured Request Formation: Finally, the self-query retriever assembles the structured request, bifurcating the semantic search term (query) from the logical conditions (filter) that streamline the document retrieval process.

### ii. Text to SQL

Considerable effort has focused on translating natural language into SQL requests, with a few notable challenges such as:

Hallucination: LLMs are prone to ‘hallucination’ of fictitious tables or fields, thus creating invalid queries. Approaches must ground these LLMs in reality, ensuring they produce valid SQL aligned with the actual database schema. 
User errors: Text-to-SQL approaches should be robust to user mis-spellings or other irregularities in the user input that could results in invalid queries.

With these challenges in mind a few tricks have emerged:

Database Description: To ground SQL queries, an LLM must be provided with an accurate description of the database. One common text-to-SQL prompt employs an idea reported in several papers: provide the LLM with a CREATE TABLE description for each table, which include column names, their types, etc followed by three example rows in a SELECT statement. 
Few-shot examples: Feeding the prompt with few-shot examples of question-query matches can improve the query generation accuracy. This can be achieved by simply appending standard static examples in the prompt to guide the agent on how it should build queries based on questions. 
Error Handling: When faced with errors, data analysts don't give up—they iterate. We can use tools like SQL agents (here) to recover from errors.

### iii. Text to SQL + Semantic

### iv. Text to Cypher


## Indexing

### i. Chunking [Link](https://www.youtube.com/watch?v=8OJC21T2SL4)
Character level Splitting - This splits the text into user defined chunk lengths. It doesn't take into account semantic meaning while chunking. It can chunk words in-between and hence not recommended .
Recursive Character Splitting - It specify series of seperator which will be used to split the docs.  Default seperators are:
"\n\n" , "\n", " ", "".  This does not split the words in-between.
Document Specific Splitting - It splits the documents based on document type like python file, html file, javascript file, etc. It can also extract images and tables from pdf.
Semantic Chunking - 
Agentic Splitting - 

### ii. Multi representation  indexing
Parent Document - In this case, based on the user query, one can retrieve the most related chunk and instead of just passing that chunk to the LLM, pass the parent document that the chunk is a part of. This helps improve the context and hence retrieval. However, what if the parent document is bigger than the context window of the LLM? We can make bigger chunks along with the smaller chunks and pass those instead of the parent to fit the context window

<img width="643" alt="Screen Shot 2025-01-17 at 4 04 27 PM" src="https://github.com/user-attachments/assets/4752e47e-e4b1-49ab-87f2-e7d3ec5bdaee" />

Dense X Retrieval - Dense Retrieval is a new method for contextual retrieval whereby the chunks are not sentences or paragraphs as we saw earlier. Rather, the authors in the below paper introduce something called “proposition”. A proposition effectively encapsulates the following:
Distinct meaning in the text. The meaning should be captured such that putting all propositions together covers the entire text in terms of semantics. Minimal, ie cannot be further split into smaller propositions. “contextualized and self contained”, meaning each proposition by itself should include all necessary context from the text.

### iii. Specialized Embedding models
In a RAG application, the responses of a large language model are augmented with relevant, up-to-date context retrieved from a vector store. Typically, similarity search is employed to retrieve such documents by comparing the user query embedding with the documents' embeddings. The effectiveness of retrieval via similarity search heavily depends on the quality of the embeddings used to represent the documents and queries. Consequently, the choice of an embedding model will significantly impact the retrieval precision, which in turn will affect the quality of the LLM responses and the RAG application as a whole. 

#### Fine Tunning
Bi-Encoders vs Cross-Encoders
Bi-Encoders produce a vector representation for a given sentence or document chunk; which is usually a single vector of a fixed dimension. Note that there’s an exception to this - ColBERT, but we won’t be covering it in this blog post. In most cases, whether the input text is a single sentence, such as a user question, or a full paragraph, like a document excerpt, as long as the input fits within the embedding model's maximum sequence length, the output will be a fixed-dimension vector. Here’s how it works: the pre-trained encoder model (usually BERT) converts the text into tokens, for each of which it has learned a vector representation during pre-training. It then applies a pooling step to average individual token representations into a single vector representation. 
CLS pooling: vector representation of the special [CLS] token (designed to model the representation for the sentence that follows it) becomes the representation for the whole sequence
Mean pooling: the average of token vector representations is returned as the representation for the whole sequence
Max pooling: the token vector representation with the largest values becomes the representation for the whole sequence

The bi in Bi-Encoder stems from the fact that documents and user queries are processed separately by two independent instances of the same encoder model. The produced vector representations can then be compared using cosine similarity: 
<img width="711" alt="Screen Shot 2025-01-17 at 6 22 31 PM" src="https://github.com/user-attachments/assets/97b525c3-1a28-40b3-a6c7-6496c9f30d8d" />


A Cross-Encoder takes in both text pieces (e.g. a user query and a document) simultaneously. It does not produce a vector representation for each of them, but instead outputs a value between 0 and 1 indicating the similarity of the input pair. 
<img width="585" alt="Screen Shot 2025-01-17 at 6 23 10 PM" src="https://github.com/user-attachments/assets/f22ccdc9-2b90-44fe-a533-21c2380b018d" />

Pre-training an embedding model
Although training steps may vary slightly from one model to another, and not all model publishers have shared their training details, the pre-training steps for Bi-Encoder embedding models generally follow a similar pattern.
The process begins with a pre-trained general-purpose encoder-style model, such as a small, ~100M-parameter pre-trained BERT. Despite the availability of larger, more advanced generative LLMs, this smaller BERT model remains a solid backbone for text embedding models today.
To fine-tune the pre-trained BERT for information retrieval, a dataset is assembled consisting of text pairs (question/answer, query/document) with a contrastive learning objective that reflects the downstream use of text embeddings. The text pairs can be positive (e.g. a question and an answer to it), and negative (a question and unrelated text). The goal is for the model to learn to bring the embeddings of positive pairs closer together in vector space while pushing the embeddings of negative pairs apart. 
The training process often has more than one step. Initially, the model is trained on a large corpus of text pairs in a weakly supervised manner. However, to achieve SOTA results on the leaderboards, a second round of contrastive training is often employed. In this second round, the model is further trained on a smaller dataset with high-quality data and particularly challenging examples from academic datasets like MSMARCO, HotpotQA, and NQ. 


#### ColBERT

### iv. Hierarchical Indexing
RAPTOR - The RAPTOR model as proposed by Stanford researchers is based on tree of document summarization at various abstraction levels ie creating a tree by summarizing clusters of text chunks for more accurate retrieval. The text summarization for retrieval augmentation captures a much larger context across different scales encompassing both thematic comprehension and granularity.
The paper claims significant performance gains by using this method of retrieving with recursive summaries. For example, “On question-answering tasks that involve complex, multi-step reasoning, we show state-of-the-art results; for example, by coupling RAPTOR retrieval with the use of GPT-4, we can improve the best performance on the QuALITY benchmark by 20% in absolute accuracy.”
<img width="869" alt="Screen Shot 2025-01-17 at 4 09 27 PM" src="https://github.com/user-attachments/assets/c9a4b689-b584-40a4-807d-1b6674dab48d" />

### v. Indexing for faster search in Vector DB
#### k-nearest neighbours - KNN (K-Nearest Neighbors) indexing is a brute-force technique used in the retrieval stage of RAG models.
Similarity Measure: A function (e.g., cosine similarity, L2 distance) that determines the “closeness” between two vectors. The retrieval process aims to identify passages/documents whose vector representations are most similar to the query vector.
Note: Use L2 distance when the magnitude of features matters (Ideal when raw values and their differences are important, like comparing user profiles based on age and income) otherwise use cosine similarity (Ideal when comparing documents based on word topics, regardless of the overall word count).
Search Process:
Query Vectorization: The user’s query is transformed into a vector using the same method applied to the corpus.
KNN Search: It retrieves the k nearest neighbour vectors from the corpus using given similarity metrics. Its time complexity is O(N log k) but using the KD tree for search will reduce the time complexity to O(log N)
k Value: The number of nearest neighbours to retrieve for each query. A higher k value provides more diverse results, while a lower k value prioritizes the most relevant ones.

#### Inverted File Vector - IVF indexing is an ANN technique used to accelerate the retrieval stage in RAG. It shares some similarities with KNN indexing but operates in a slightly different way.

Core Concepts:

Clustering: The vector space is partitioned into clusters using techniques like k-means clustering. Each cluster has a centroid, which represents the center of that cluster in the vector space.
Inverted File: An inverted file data structure (Just like a Python dictionary) is created. This file maps each centroid to a list of data point IDs (passages/documents) that belong to the corresponding cluster.
Search Process:

Nearest Centroid Search: The IVF index efficiently searches for the nearest centroid (the cluster centroid vector most similar to the query vector) based on a similarity measure (often cosine similarity).
Refined Search: Within the cluster identified by the nearest centroid, a smaller number of nearest neighbours (data points) are retrieved using a more expensive distance metric (like L2 distance). This step refines the search within the most promising cluster.

#### Locality Sensitive Hashing
LSH indexing serves to expedite the retrieval process. Unlike the previously discussed methods (KNN, IVF), LSH focuses on mapping similar data points to the same “buckets” with a high probability, even though it might not guarantee the closest neighbours.

Core Concepts:

Hash Functions: LSH utilizes a family of hash functions that map similar data points (represented as vectors) to the same “hash bucket” with a high probability. However, dissimilar data points might also collide in the same bucket.
Similarity Measure: A function (like cosine similarity) determines the “closeness” between two vectors. The LSH functions are designed to map similar vectors (based on the similarity measure) to the same bucket with a higher probability compared to dissimilar vectors.
Multiple Hash Tables: LSH often uses multiple hash tables, each employing a different LSH function. This approach increases the likelihood of finding similar items even if they collide in one table, as they might be separated in another.
Search Process Steps:

LSH Hashing: The query vector is hashed using each hash function in the LSH family, resulting in a set of hash codes (bucket indices) for each table.
Candidate Retrieval: Based on the generated hash codes, documents that share at least one hash code (i.e., potentially similar based on the LSH function) with the query in any of the tables are considered candidate matches.

#### Random Projection - 
RP indexing projects high-dimensional data (text vectors) into a lower-dimensional space while attempting to preserve similarity relationships.

Core Concepts:

Dimensionality Reduction: RP aims to reduce the dimensionality of textual data represented as vectors. High-dimensional vectors can be computationally expensive to compare.
Random Projection Matrix: A random matrix is generated, with each element containing random values (often following a Gaussian distribution). The size of this matrix determines the target lower dimension.
Similarity Preservation: The goal is to project data points (vectors) onto a lower-dimensional space in a way that retains their relative similarity as much as possible in the high-dimensional space (Binary form). Just like SVM, when the hyperplane normal vector produces a +ve dot-product with another vector, we encode it as 1 else 0. This allows for efficient search in the lower-dimensional space while still capturing relevant connections.

Search Process Steps:

Random Projection: Both the query vector and the document vectors in the corpus are projected onto the lower-dimensional space using the pre-computed random projection matrix. This results in lower-dimensional representations of the data (binary vector).
Search in Lower Dimension: Efficient search algorithms (Hamming distance to find the closest match) are used in the lower-dimensional space to find documents whose projected vectors are most similar to the projected query vector. This search can be faster due to the reduced dimensionality.

#### Product Quantization - PQ indexing is used to accelerate the search process and reduce memory footprint.

Core Concepts:

Vector Decomposition: High-dimensional query and document vectors are decomposed into smaller sub-vectors representing lower-dimensional subspaces. This effectively breaks down the complex vector into simpler pieces.
Codebook Creation: For each subspace, a codebook is created using techniques like k-means clustering. This codebook contains a set of representative centroids, each representing a group of similar sub-vectors.
Encoding: Each sub-vector is then “encoded” by identifying the closest centroid in its corresponding codebook. This encoding process assigns an index to the sub-vector based on its closest centroid.

Search Process Steps:

Vector Decomposition: The query vector is decomposed into sub-vectors according to the pre-defined subspaces.
Subspace Encoding: Each sub-vector is encoded by finding its closest centroid in the corresponding codebook, resulting in a set of indices representing the encoded sub-vectors.
Approximate Distance Calculation: Using the encoded sub-vector indices from both the query and document vectors, an efficient distance metric is applied to estimate the similarity between the two vectors.

#### Hierarchical Navigable Small World
HNSW excels at finding data points in a large collection that are most similar to a given query, but it might not pinpoint the absolute exact closest neighbour. This trade-off between perfect accuracy and retrieval speed makes HNSW ideal for applications like RAG indexing, where returning highly relevant information quickly is crucial.

Core Concepts:

Navigable Small World Graphs: HNSW builds upon the concept of navigable small world graphs. Imagine a social network where everyone is connected to a few close friends but also has a few random connections to people further away in the network. This allows for efficient navigation — you can usually reach any person within a small number of hops by strategically following close and distant connections. HNSW translates this concept into a graph structure where data points (vectors) are nodes, and connections represent their similarity.
Skip list: Similar to the Linked list data structure, HNSW introduces a hierarchical structure to the graph. This means there are multiple layers, each with a different “grain size” for connections(Skip Connection). The top layer has far fewer connections but spans larger distances in the vector space, allowing for quick exploration. Lower layers have more connections but focus on finer-grained similarity. This hierarchy enables efficient search — the algorithm starts at the top layer for a broad initial search and then progressively refines in lower layers to find the nearest neighbours. But for this, we need to build a sorted skip list.
Search Process:

Top-Layer Exploration: HNSW leverages the long-distance connections in the top layer to identify a small set of potentially promising nodes (candidate nearest neighbours).
Hierarchical Descent: The algorithm iteratively explores these candidates in lower layers, using shorter-distance connections to refine the search and get closer to the true nearest neighbours.
Selection: Throughout the search, a pre-defined number of nearest neighbours are selected based on a distance threshold or other criteria.








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

