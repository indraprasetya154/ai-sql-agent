import os
from typing import List, Any, TypedDict
from dotenv import load_dotenv
from langchain_google_genai import HarmCategory, HarmBlockThreshold
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
import psycopg2
from .document_sql import sql_documents, sql_uuids
from .config import get_config

# Load environment variables
load_dotenv()
config = get_config()

# Type definitions for state
class AgentState(TypedDict):
    user_input: str
    relevant_docs: List[Any]
    sql_query: str
    query_result: List[Any]
    final_response: str

# Initialize LLM
def initialize_llm():
    google_api_key = config["GOOGLE_API_KEY"]

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash",
        temperature=0.2,
        safety_settings=safety_settings,
        api_key=google_api_key
    )
    
    return llm

# Initialize embeddings
def initialize_embeddings():
    google_api_key = config["GOOGLE_API_KEY"]
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=google_api_key
    )
    
    return embeddings

# Initialize vector store
def initialize_vector_store():
    embeddings = get_embeddings()
    
    d = len(embeddings.embed_query("hello"))
    index = faiss.IndexFlatL2(d)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    # Add documents to vector store
    vector_store.add_documents(documents=sql_documents, ids=sql_uuids)
    print("Documents successfully added to vector store!")

    return vector_store

# Node functions for the graph
def retrieve_relevant_docs(state: AgentState) -> AgentState:
    """Retrieve relevant documents based on user input."""
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 4}
    )

    relevant_docs = retriever.invoke(state["user_input"])
    return {"relevant_docs": relevant_docs}

def generate_sql_query(state: AgentState) -> AgentState:
    """Generate SQL query based on relevant documents."""
    llm = get_llm()

    messages = [
        SystemMessage(content=f"""Anda adalah AI yang bertugas memvalidasi dan menjelaskan SQL query untuk menjawab pertanyaan pengguna.

        Tugas Anda:
        1. Periksa apakah SQL query yang ditemukan dalam dokumen RAG dapat menjawab pertanyaan pengguna
        2. Informasikan ID dan SQL query yang sesuai
        3. Jelaskan mengapa SQL query tersebut sesuai atau tidak sesuai untuk pertanyaan tersebut
        4. Jelaskan bagaimana SQL query tersebut bekerja dan data apa yang akan dihasilkan
        5. Jika ada beberapa SQL query yang relevan, jelaskan mana yang paling sesuai

        Berikan hanya dalam response dalam bentuk SQL string, agar langsung dieksekusi, tanpa perlu memberikan penjelasan

        Jika membutuhkan penggabungan query, lakukan penggabungan query dulu agar dapat diekesekusi dalam satu query
        gunakan SELECT (SATU-KOLOM), HINDARI menggunakan SELECT *

        Tidak perlu menggunakan ```sql```.

        Dokumen RAG: {state["relevant_docs"]}
        """),
        HumanMessage(content=state["user_input"]),
    ]

    response = llm.invoke(messages)
    sql_query = response.content.strip()

    return {"sql_query": sql_query}

def execute_query(state: AgentState) -> AgentState:
    """Execute the generated SQL query."""
    try:
        conn = psycopg2.connect(
            host=config["DB_HOST"],
            port=config["DB_PORT"],
            user=config["DB_USER"],
            password=config["DB_PASSWORD"],
            database=config["DB_DB"]
        )
        cur = conn.cursor()
        cur.execute(state["sql_query"])
        query_result = cur.fetchall()
        conn.commit()
        cur.close()
        conn.close()

        print("\n=== SQL Query Result ===")
        print(query_result)

        return {"query_result": query_result}
    except Exception as e:
        print(f"❌ Error executing SQL query: {e}")
        return {"query_result": None}  # Return None to indicate failure

def generate_final_response(state: AgentState) -> AgentState:
    """Generate the final response based on the query results."""
    llm = get_llm()
    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 4}
    )

    retriever_result_query = retriever.invoke(str(state["query_result"]))

    messages = [
        SystemMessage(content=f"""Validasi Jawaban dari Output query SQL yang telah dijalankan sebelumnya, jika memang valid, berikan output yang mudah dimengerti

        Jika memang bisa ditotalkan, totalkan hasil query

        Buatkan response dalam bahasa manusia

        Query yang dijalankan : {state["sql_query"]}
        Document RAG : {retriever_result_query}
        Ini adalah output dari hasil query: {state["query_result"]}
        """),
        HumanMessage(content=state["user_input"]),
    ]

    response = llm.invoke(messages)
    final_response = response.content

    # Handle empty or invalid query results
    if not state.get("query_result"):
        final_response = "Maaf, saya tidak dapat menemukan data yang sesuai dengan permintaan Anda."

    return {"final_response": final_response}

# For older versions of LangGraph that use conditional edges differently
def check_result(state: AgentState):
    """Check if query result exists and return next node."""
    if not state.get("query_result"):
        return "generate_response"  # Fallback to generate_response even if query fails
    else:
        return "generate_response"

# Global variables for singletons
_llm = None
_embeddings = None
_vector_store = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = initialize_llm()
    return _llm

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = initialize_embeddings()
    return _embeddings

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = initialize_vector_store()
    return _vector_store

# Build the graph
def build_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("retrieve_docs", retrieve_relevant_docs)
    workflow.add_node("generate_sql", generate_sql_query)
    workflow.add_node("execute_query", execute_query)
    workflow.add_node("generate_response", generate_final_response)

    # Add edges - using simpler approach
    workflow.add_edge("retrieve_docs", "generate_sql")
    workflow.add_edge("generate_sql", "execute_query")
    # Based on execute_query result, go to either generate_response or retry
    workflow.add_conditional_edges(
        "execute_query",
        check_result,
        {
            "retrieve_docs": "retrieve_docs",
            "generate_response": "generate_response"
        }
    )
    workflow.add_edge("generate_response", END)

    # Set entry point
    workflow.set_entry_point("retrieve_docs")

    return workflow.compile()

# Main function to run the agent
def ai_sql_agent(user_input: str):
    # Build the graph
    graph = build_graph()

    # Run the graph
    try:
        result = graph.invoke({"user_input": user_input})
        return result["final_response"]
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        return "Maaf saya belum menyimpan informasi tersebut saat ini, silahkan coba lagi nanti"
