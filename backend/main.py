from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai.errors import ClientError
import os

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        result = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=req.message,
        )
        return {"response": result.text}
    except ClientError as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            raise HTTPException(status_code=429, detail="API quota exhausted")
        raise
