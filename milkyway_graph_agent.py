
import os
import sqlite3
import requests
from bs4 import BeautifulSoup
from git import Repo
import json
from openai import OpenAI
import shutil

from langchain_community.vectorstores import SQLiteVSS


openai_base_url = "https://gds.paypalinc.com/jupiter/ai4tech-gemini-service/openai/v1/"
openai_api_key = "xxx"

openai_client = OpenAI(base_url=openai_base_url, api_key=openai_api_key)

from langchain_community.embeddings import SentenceTransformerEmbeddings
embedding_func = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = SQLiteVSS(
    db_file="rag/milkyway_graph_rag_vss.db",
    table="milkyway_graph_rag",
    connection=None,
    embedding=embedding_func,
)

message = ("account_lnk_by_address_v2 这个类的具体逻辑帮我概括一下")
print(message)

template = """请根据提供的文档片段以及用户问题生成答案, 如果有可能, 请给出所引用的代码的文件名路径. \n ###包含的是文档片段:  ###{rag_context}###\n 问题: {question}"""

rag_context = vector_store.similarity_search(message, k=8)
print(rag_context)
context_str = ";".join([d.page_content for d in rag_context])
chat_to_agent_str = template.format_map({"rag_context": context_str, "question": message})

messages = [
    {
        "role": "user",
        "content": chat_to_agent_str,
    },
]

params = dict(
    model="gemini-1.5-flash-001",
    messages=messages,
    stream=False,
    temperature=0,
)
response = openai_client.chat.completions.create(**params)
reply = response.choices[0].message.content
print(reply)