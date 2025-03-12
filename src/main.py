from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .ai_sql import ai_sql_agent
import json

app = FastAPI()

class QueryRequest(BaseModel):
    message: str

class QueryResponse(BaseModel):
    message: str
    answer: str

@app.post("/query")
async def sql_agent(request: QueryRequest):
    try:
        result = ai_sql_agent(request.message)  # Hasilnya berupa string JSON
        return QueryResponse(message=request.message, answer=result)

    except (json.JSONDecodeError, KeyError) as e:
        raise HTTPException(status_code=500, detail=f"Invalid response format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
