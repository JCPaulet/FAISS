#In order to run the streamlit app based on the code below you have to run the following commands in the command prompt:

#cd "C:\Users\jcpan\.cursor-tutor\.venv\"
#.venv\Scripts\activate
#streamlit run CB_RAG_LANGGRAPH_V4.2.py 
#version 2 :
# Summary of Changes for V3:
    #Caching with @st.cache_data: The load_documents_and_embeddings function is decorated with @st.cache_data to ensure that the documents and their embeddings are only loaded and computed once.

     #Streamlit Application:

        #The state dictionary captures the user input (the question).
        #The Retrieve button triggers the vectordb.max_marginal_relevance_search function to retrieve relevant documents.
        #The Generate button triggers a simplified generation logic to display the concatenated content of the retrieved documents.
        #This setup ensures that the embeddings are computed only once, and subsequent questions use the cached embeddings, improving the efficiency of the Streamlit application.
    #
    #
# Summary of changes from V4.1:introduction of  ".add" to messages to add persistence as it seems that :memory: is not working in V3 and using a method to split chunks based on their semantic similarity (https://python.langchain.com/docs/how_to/semantic-chunker/) 
import streamlit as st
from pprint import pprint
import os
from uuid import uuid4
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain import hub
from langchain.schema import Document
from langchain_core.documents import Document as CoreDocument
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import List, TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
import faiss
import numpy as np
from langsmith import traceable
import chromadb
import SentenceTransformerEmbeddingFunction as embedding_functions #importing the code of the class "SentenceTransformerEmbeddingFunction" which I saved in the same folder as this file.
from langchain_experimental.text_splitter import SemanticChunker
from chromadb import Client
from langchain.embeddings.base import Embeddings
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
#from langchain_experimental.text_splitter import SemanticChunker
import sys

# Override sqlite3 with pysqlite3
if 'sqlite3' in sys.modules:
    del sys.modules['sqlite3']

import pysqlite3 as sqlite3
sys.modules['sqlite3'] = sqlite3

 # Initialize embeddings and specifying a model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


load_dotenv()  # Loads variables from .env file into environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo-1106", openai_api_key=OPENAI_API_KEY, temperature=0)

#load FAISS vector db
new_vector_store = FAISS.load_local(folder_path="main",
                                    index_name="faiss_index",
                                    embeddings=embeddings, 
                                    allow_dangerous_deserialization=True
                                    )

# Cache the function to load documents and compute embeddings
#@st.cache_data(show_spinner=False)


multiply_query_prompt = ChatPromptTemplate(
        input_variables=['query'],
        messages=[HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['query'],
                template=(
            "You are a helpful Erasmus + programme guide expert. Your users are asking questions about the Erasmus + programme guide. "
            "Suggest up to five additional related questions to help them find the information they need, for the provided question. "
            "Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic."
            "Make sure they are complete questions, and that they are related to the original question."
            "Output one question per line. Do not number the questions."
            "content: {query}"
                )
                     )
        )]
    )
multiply_query= multiply_query_prompt | llm | StrOutputParser()  

retrieval_grader_prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question.
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question}
    If the document contains keywords related to the user question, grade it as relevant.
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["question", "document"],
)
retrieval_grader = retrieval_grader_prompt | llm | JsonOutputParser()

# Prompt template for the main task
rag_prompt = ChatPromptTemplate(
    input_variables=['context', 'question','messages'],
    messages=[HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=['context', 'question','messages'],
            template=(
                "You are an assistant for question-answering tasks. Use only the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, just say that you don't know. Use 25 sentences maximum.\n"
                "Question: {question} \n"
                "Context: {context} \n"
                "Make sure to include the source and page number in your answer when relevant.\n"
                "Answer:"
            )
        )
    )]
)
rag_chain = rag_prompt | llm | StrOutputParser()

# Grading and rewriting prompts
hallucination_grader_prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts.
    Here are the facts:
    {documents}
    Here is the answer: {generation}
    Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts.
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "documents"],
)
hallucination_grader = hallucination_grader_prompt | llm | JsonOutputParser()

answer_grader_prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is useful to resolve a question.
    Here is the answer:
    {generation}
    Here is the question: {question}
    Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question.
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "question"],
)
answer_grader = answer_grader_prompt | llm | JsonOutputParser()

re_write_prompt = PromptTemplate(
    template="""You a question re-writer that converts an input question to a better version that is optimized
    for vectorstore retrieval. Look at the initial and formulate an improved question.
    Here is the initial question: \n\n {question}. Improved question with no preamble: \n """,
    input_variables=["question"],
)
question_rewriter = re_write_prompt | llm | StrOutputParser()


# Define GraphState to use CustomDocument
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[dict]
    
    def __repr__(self):
        return self.name

# Define the retrieve function

@traceable
def retrieve(state: GraphState):
    print('retrieve')
    question = state["question"]
    multiple_queries=multiply_query.invoke({"query": question})

    # Split the string into a list using the '-' as the delimiter
    queries_list = multiple_queries.split('\n')

    # Remove any empty strings that may result from the split
    queries_list = [sentence.replace('-', '').strip() for sentence in queries_list if sentence.strip()]
    print(queries_list)

    queries = [question] + queries_list
    #retrieval using FAISS
    results = new_vector_store.similarity_search( queries,
    k=3,)

    retrieved_documents = results['documents']
    retrieved_ids = results['ids']
    retrieved_metadatas = results['metadatas']
    unique_documents = set()
    unique_doc_indices = {}  # To keep track of original indices

    for i, documents in enumerate(retrieved_documents):
        for j, document in enumerate(documents):
            if document not in unique_documents:
                unique_documents.add(document)
            unique_doc_indices[document] = (i, j)

    unique_documents = list(unique_documents)

    pairs = []
    metadatas = []
    for doc in unique_documents:
        pairs.append([question, doc])
        i, j = unique_doc_indices[doc]
        metadatas.append(retrieved_metadatas[i][j])
    
    from sentence_transformers import CrossEncoder
    cross_encoder = CrossEncoder('sentence-transformers/all-mpnet-base-v2')
   
# Get the scores for each pair
    scores = cross_encoder.predict(pairs)

# Combine scores with their corresponding pairs and metadatas
    scored_pairs = list(zip(scores, pairs, metadatas))

# Sort the scored_pairs by score in descending order
    scored_pairs_sorted = sorted(scored_pairs, key=lambda x: x[0], reverse=True)

# Keep only the top 7 pairs
    top_7_pairs = scored_pairs_sorted[:7]
   # Create CustomDocument objects
    context = []
    for score, pair, metadata in top_7_pairs:
        _, document = pair
        custom_doc = CustomDocument(page_content=document, metadata=metadata)
        context.append(custom_doc)

    return {"documents": [doc.to_dict() for doc in context], "question": question}
    
   
@traceable
# Define the generate function
def generate(state: GraphState):
    print('generate')
    question = state["question"]
    documents = [CustomDocument(**doc) for doc in state["documents"]]
    context = "\n\n".join(
        f"Content: {doc.page_content}\nSource: {doc.metadata.get('source', 'N/A')}\nPage: {doc.metadata.get('page', 'N/A')}"
        for doc in documents
    )
    #print(context)
    generation = rag_chain.invoke({"context": context, "question": question})
    return {"documents": [doc.to_dict() for doc in documents], "question": question, "generation": generation}

@traceable
# Define the grade_documents function
def grade_documents(state: GraphState):
    print('grade')
    question = state["question"]
    documents = [CustomDocument(**doc) for doc in state["documents"]]
    #print(documents)
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        if score['score'] == "yes":
            filtered_docs.append(d)
    return {"documents": [doc.to_dict() for doc in filtered_docs], "question": question}

@traceable
def transform_query(state: GraphState):
    question = state["question"]
    documents = state["documents"]
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}
@traceable
def decide_to_generate(state: GraphState):
    filtered_documents = state["documents"]
    if not filtered_documents:
        return "transform_query"
    return "generate"
@traceable
def grade_generation_v_documents_and_question(state: GraphState):
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    if score['score'] == "yes":
        score = answer_grader.invoke({"question": question, "generation": generation})
        if score['score'] == "yes":
            return "useful"
        return "not useful"
    return "not supported"

# Initialize workflow
#initial_state = {"question": "", "generation": "", "documents": []}

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"transform_query": "transform_query", "generate": "generate"},
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {"not supported": "generate", "useful": END, "not useful": "transform_query"},
)
with SqliteSaver.from_conn_string(":memory:") as checkpointer:
    appli = workflow.compile(checkpointer=checkpointer)
    
# Streamlit UI setup
def generate_response(question):
    inputs = {"question": question}
    config = {"configurable": {"thread_id":"2"}}
    try:
        for output in appli.stream(inputs, config=config):
            for key, value in output.items():
                pprint(f"Node '{key}':")
            pprint("\n---\n")
        return value["generation"]
    except KeyError as e:
        print(f"KeyError: {e}")
        return "An error occurred."
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred."

def write_message(role, content, save=True):
    if save:
        st.session_state.messages.append({"role": role, "content": content})
    with st.chat_message(role):
        st.markdown(content)

st.set_page_config("E+ Prog. Guide Chatbot", page_icon=":notebook:")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm the Erasmus + programme guide Chatbot! How can I help you?"},
    ]

def handle_submit(message):
    with st.spinner('Thinking...'):
        response = generate_response(message)
        write_message('assistant', response)

for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

if question := st.chat_input("What is your question?... Type it here"):
    write_message('user', question)
    handle_submit(question)
