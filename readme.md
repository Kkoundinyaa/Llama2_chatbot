In this repository we are going to build a chatbot which works on the idea of Smart Knowledge Base and makes use of Natural Language Processing (NLP) and Large Language Models (LLMs). This chatbot is a question answer chatbot that let’s us chat with our documents. Let us understand the key components behind the working of this chatbot.

What is a Smart Knowledge Base? 
Smart Knowledge Base is an advanced digital repository which uses Artificial Intelligence and Natural Language Processing to organize, curate and provide dynamic access to a large collection of data. It can understand and respond to user queries, while continuously and updating its content to stay relevant. 
What are LLMs?
LLM stands for Large Language Model which is a concept in Artificial Intelligence which is designed to understand and give human like response to the given input. It can have sophisticated conversations, generate content, and perform various language related tasks.
What is Natural Language Processing?
Natural Language Processing (NLP) is a branch of artificial intelligence that focuses on enabling computers to understand, interpret, and generate human language, allowing for tasks like language translation, sentiment analysis, and text generation.

Step-1:
Create a folder with your desired name and open it in the command prompt or any IDE and create a virtual environment and then git clone this repository into your system locally using the following command:

We also need to download the Llama-2 model from HuggingFace into the same folder that we created. Once you clone the repository then identify the requirements.txt file and install all the dependencies mentioned in the text file using pip install “required package name”.

Step-2:
Once you clone the repository you will notice two different python files ingest.py and main.py, ‘ingest.py’ is used to identify the data and store in a vector database, main.py is the code that is required to develop the chatbot.
Open ingest.py and import all the modules from the langchain package. Now we create a data path i.e., it is a directory that holds the documents (this data can be of any form pdf, csv, docx).
DATA_PATH = "DATA/"

Next subsequently we also create a path for our vector database where the vectordb is created and is persisted from.
DB_FAISS_PATH = "vectorstores/db_faiss"

Step-3:
Now we will convert our data into vectors/embeddings with the help of HuggingFaceEmbeddings which are sentence transformers* in general. Embeddings are essentially vector form of our input data, input data can be of any format text, pictures, etc. The converted embeddings are stored in the vector database and saved locally so that we can persist it later on in our code. 
db = FAISS.from_documents(texts, embeddings)
db.save_local(DB_FAISS_PATH)

Step-4:
We now move to main.py and import all the packages from langchain module and import chainlit as cl. We then create a custom prompt for the chatbot so that it does not create it’s own responses.
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
  
We then start writing respective functions for loading the llm, creating the chain, creating the bot and finally one function for llm response.

Step-5:
In our Smart Knowledge Base we are going to use Llama-2 as our llm but to load the llm in a much easier way we are making use of the CTransformers package in python which helps in loading of the Llama-2 model.
def load_llm():
    llm = CTransformers(
        model= "llama-2-7b-chat.ggmlv3.q3_K_M.bin",
        model_type= "llama",
        max_new_tokens = 128,
        temperature = 0.5)

Once the code is ready and good to go open the terminal and run your code!



