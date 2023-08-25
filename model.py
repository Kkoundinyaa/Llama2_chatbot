from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = "vectorstores/db_faiss"

custom_prompt_template = """Use this as a custom prompt template and use the information to answer the user's question.
 If you don't know the answer just say I don't know the answer do not try to make up your own answer.
 
 Context: {}
 Question: {question}
 
 Only returns the helpful answer below and nothing else
 Helpful Answer:
 """

def set_custom_prompt():
    prompt = PromptTemplate(template= custom_prompt_template, input_variables=['Context', 'Question'])

    return prompt

def load_llm():
    llm = CTransformers(
        model= "llama-2-7b-chat.ggmlv3.q3_K_M.bin",
        model_type= "llama",
        max_new_tokens = 128,
        temperature = 0.5
    )

    return llm

def retrival_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents = True,
        chain_type_kwargs= {'prompt': prompt}
    )

    return qa_chain

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', 
    model_kwargs = {'device' : 'cpu'})

    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrival_qa_chain(llm, qa_prompt, db)

    return qa


def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})

    return response


### New to me Chainlit ###
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content = "The bot is starting.....")
    await msg.send()
    msg.content = "Hi, Welcome to the ChatBot. What is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.set("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer = True, answer_prefix_tokens = ["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks = [cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += f"\nNo Sources Found"

    await cl.Message (content=answer).send()
