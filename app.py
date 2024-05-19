import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage, Document
from langchain_community.vectorstores import FAISS
from fastapi import FastAPI, Request, Form, Response, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import LLMChain
import json
import logging
from fastapi.logger import logger
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OpenAI API key not found in environment variables")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

prompt_template = """You are a vet doctor and an expert in analyzing dog's health.
Answer the question based only on the following context, which can include text, images and tables:
{context}
Question: {question}
Don't answer if you are not sure and decline to answer and say "Sorry, I don't have much information about it."
Just return the helpful answer in as much as detailed possible.
Answer:
"""

prompt = PromptTemplate.from_template(prompt_template)
llm = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key, max_tokens=1024)
qa_chain = LLMChain(llm=llm, prompt=prompt)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

logging.basicConfig(level=logging.DEBUG)

@app.post("/get_answer")
async def get_answer(question: str = Form(...)):
    try:
        relevant_docs = db.similarity_search(question)
        context = ""
        relevant_images = []
        for d in relevant_docs:
            if d.metadata['type'] == 'text':
                context += '[text]' + d.metadata['original_content']
            elif d.metadata['type'] == 'table':
                context += '[table]' + d.metadata['original_content']
            elif d.metadata['type'] == 'image':
                context += '[image]' + d.page_content
                relevant_images.append(d.metadata['original_content'])
        result = qa_chain.run({'context': context, 'question': question})
        return JSONResponse({"relevant_images": relevant_images[0], "result": result})
    except Exception as e:
        logger.exception("An error occurred: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)
