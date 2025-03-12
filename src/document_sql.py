from uuid import uuid4
from langchain.docstore.document import Document

# Define SQL query templates
sql_documents = [
    Document(
        page_content="""Question: How many orders were placed in {year}?
        SQL Query:
        SELECT total FROM total_orders WHERE year = {year};""",
        metadata={"source": "sql"},
    ),
    Document(
        page_content="""Question: How many users registered in {year}?
        SQL Query:
        SELECT total FROM total_users WHERE year = {year};""",
        metadata={"source": "sql"},
    ),
]

sql_uuids = [str(uuid4()) for _ in range(len(sql_documents))]
