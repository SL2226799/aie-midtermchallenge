import os
from typing import List
from operator import itemgetter

from openai import AsyncOpenAI  # importing openai for API usage
from langchain_openai import OpenAIEmbeddings # Importing OpenAI Embedings from the text-embedding-3-small model

from langchain_community.vectorstores import FAISS # Importing FAISS to power our vector store

from langchain.document_loaders import PyMuPDFLoader # Importing PyMuPDFLoader to load our pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter # Importing RecursiveCharacterTextSplitter for splitting text into chunks
from langchain.prompts import ChatPromptTemplate # For creating our prompt
from langchain_openai import ChatOpenAI # Using Chat model gpt-3.5-turbo with a temperature of 0 as our LLM
from langchain_core.output_parsers import StrOutputParser # for LCEL
from langchain_core.runnables import RunnablePassthrough #For LCEL
from langchain.retrievers import MultiQueryRetriever # Importing an advanced retriever from langchain
from langchain import hub #importing custom prompt from the hub
from langchain.chains.combine_documents import create_stuff_documents_chain # For combining documents
from langchain.chains import create_retrieval_chain 


from ragas.testset.generator import TestsetGenerator # For RAGAS Evaluation
from ragas.testset.evolutions import simple, reasoning, multi_context # For assorted question types
from datasets import Dataset  # Creating Hugging Face dataset for use in the Ragas library
from ragas import evaluate 
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_recall,
    context_precision,
)


import chainlit as cl  # importing chainlit for our app





def process_file():
    loader = PyMuPDFLoader("https://d18rn0p25nwr6d.cloudfront.net/CIK-0001045810/1cbe8fe7-e08a-46e3-8dcc-b429fc06c1a4.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    return documents, chunks



@cl.on_chat_start
async def on_chat_start():

    msg = cl.Message(
        content=f"Processing `NVIDIA 10k Filings RAG`...  Wait till processing is complete to ask questions..... Please be patient.", disable_feedback=True
    )
    await msg.send()

    # load and split the file into chunks
    documents, texts = process_file()
    print(texts[0])

    # Initialize our embedding model to be text-embedding-3-small
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Set up our FAISS-powered vector store, and create a retriever.
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever()

    #Set up prompt
    #prompt = get_prompt()

    # Initialize model
    openai_chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # get prompt from hub
    retrieval_qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    #create stuff document chain
    document_chain = create_stuff_documents_chain(openai_chat_model, retrieval_qa_prompt)

    # modifying retriever with an advanced one
    advanced_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=openai_chat_model)

    #creating retrieval chain
    retrieval_chain = create_retrieval_chain(advanced_retriever, document_chain)

    # Let the user know that the system is ready
    msg.content = f"Processing `NVIDIA 10k Filings RAG` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("retrieval_chain", retrieval_chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("retrieval_chain")  

    print(message.content)

    res = await chain.ainvoke({"input" : message.content})
    answer = res["answer"]
    
    await cl.Message(content=answer).send()
