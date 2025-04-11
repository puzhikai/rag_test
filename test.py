
from dolores.apps import ChatApp
from dolores.core.utils import FileUtils
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import GitLoader

from langchain_community.vectorstores import SQLiteVSS

import os
import sqlite3
import requests
from bs4 import BeautifulSoup
from git import Repo
import json
from openai import OpenAI
import shutil



repo_url = "git@github.paypal.com:edp-aiml/unified-graph-parent.git"
repo_dir = "./unified-graph-parent-repo"
if os.path.exists(repo_dir):
    shutil.rmtree(repo_dir)
Repo.clone_from(repo_url, repo_dir)

docs = []
for root, _, files in os.walk(repo_dir):
    for file in files:
        if file.endswith((".py", ".md", ".txt")):
            path = os.path.join(root, file)
            loader = TextLoader(path)
            docs.extend(loader.load())

for doc in docs:
    file_path = doc.metadata.get('file_path', '')
    doc.metadata['filename'] = os.path.basename(file_path)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks_git = text_splitter.split_documents(docs)
texts_git = [c.page_content for c in chunks_git]




url1 = "https://tinkerpop.apache.org/docs/current/reference/#gremlin-language"
url2 = "https://paypal.atlassian.net/wiki/spaces/~yulizhou/pages/534282682/Basic+Gremlin+101"
url3 = "https://paypal.atlassian.net/wiki/spaces/RTI/pages/629377450/Milkyway+Graph+Studio"

response = requests.get(url1)
soup1 = BeautifulSoup(response.content, "html.parser")
text1 = soup1.get_text()

response = requests.get(url2)
soup2 = BeautifulSoup(response.content, "html.parser")
text2 = soup2.get_text()

response = requests.get(url3)
soup3 = BeautifulSoup(response.content, "html.parser")
text3 = soup3.get_text()



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(text1) + text_splitter.split_text(text2) + text_splitter.split_text(text3) + texts_git
embedding_func = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = SQLiteVSS.from_texts(texts=chunks, embedding=embedding_func, table="test", db_file="vss.db",overwrite=True)



openai_base_url = "https://gds.paypalinc.com/jupiter/ai4tech-gemini-service/openai/v1/"
openai_api_key = "xxx"

openai_client = OpenAI(base_url=openai_base_url, api_key=openai_api_key)



async def chat(client: Client, show_drawer: bool = True):
    # ----------------------------------- Start: Event Handlers ----------------------------------- #
    async def send() -> None:
        message = str(user_text.value).strip()
        print(message)
        #template = """你是一位问答助手, 你的任务是根据###中间的context回答问题, 请准确回答问题, 不要健谈, 如果提供的文本信息结合大模型训练数据都无法回答问题, 请直接回复"提供的文本无法回答问题", 我相信你能做的很好. ###\n{context}###\n 问题: {question} """
        template = """请根据提供的文档片段以及用户问题生成答案. \n ###包含的是文档片段:  ###{rag_context}###\n 问题: {question} """

        rag_context = vector_store.similarity_search(message, k=4)
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

        #await chat_to_agent(chat_to_agent_str)

   
