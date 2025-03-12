import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain_google_genai import HarmCategory, HarmBlockThreshold
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from .document_ddl import ddl_documents, ddl_uuids
from .document_sql import sql_documents, sql_uuids
import json
import psycopg2

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

# Initialize VertexAI embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=google_api_key)

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    temperature=0.2,
    safety_settings=safety_settings,
    api_key=google_api_key
)

d = len(embeddings.embed_query("hello"))

# FAISS index setup
index = faiss.IndexFlatL2(d)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

vector_store.add_documents(
    documents=sql_documents, ids=sql_uuids)
print("Dokumen dari sumber 'documentation' berhasil ditambahkan ke vector store!")
print(d)

def ai_sql_agent(user_input: str):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 4}
    )

    retriever_result = retriever.invoke(user_input)
    messages = [
        ("system", f"""Anda adalah AI yang bertugas memvalidasi dan menjelaskan SQL query untuk menjawab pertanyaan pengguna.

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

        Dokumen RAG: {retriever_result}
        """),
        ("human", f"{user_input}"),
    ]

    try:
        llm_verify = llm.invoke(messages)
        print("\n=== SQL Query Validation ===")
        print(llm_verify.content)

        sql_query = llm_verify.content.strip()
        sql_result = execute_sql_query(sql_query)

        retriever_result_query = retriever.invoke(str(sql_result))
        end_result_messages = [
            ("system", f"""Validasi Jawaban dari Output query SQL yang telah dijalankan sebelumnya, jika memang valid, berikan output yang mudah dimengerti

            Jika memang bisa ditotalkan, totalkan hasil query

            Query yang dijalankan : {sql_query}
            Document RAG : {retriever_result_query}
            Ini adalah output dari hasil query: {sql_result}
            """),
            ("human", f"{user_input}"),
        ]

        end_llm_verify = llm.invoke(end_result_messages)
        print("\n=== SQL Query Validation ===")
        print(end_llm_verify.content)

        result = end_llm_verify.content

        return result

    except Exception as e:
        print(f"❌ Error during LLM validation: {e}")
        return "Maaf saya belum menyimpan informasi terebut saat ini, silahkan coba lagi nanti"

def execute_sql_query(sql_query):
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_DB")
    )
    cur = conn.cursor()
    cur.execute(sql_query)
    query_result = cur.fetchall()
    conn.commit()
    print("\n=== SQL Query Result ===")
    print(query_result)

    cur.close()
    conn.close()
    return query_result
