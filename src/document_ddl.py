from uuid import uuid4
from langchain.docstore.document import Document

# Define DDL documents
ddl_documents = [
    Document(
        page_content="""CREATE TABLE IF NOT EXISTS total_users (
            year INTEGER PRIMARY KEY,
            total INTEGER NOT NULL
        );""",
        metadata={"source": "ddl"},
    ),
    Document(
        page_content="""CREATE TABLE IF NOT EXISTS total_orders (
            year INTEGER PRIMARY KEY,
            total INTEGER NOT NULL
        );""",
        metadata={"source": "ddl"},
    ),
]

ddl_uuids = [str(uuid4()) for _ in range(len(ddl_documents))]
