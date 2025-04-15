
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter
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
        if file.endswith((".py", ".md", ".txt", ".java")) or "resources" in os.path.abspath(file):
            path = os.path.join(root, file)
            loader = TextLoader(path)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata['filename'] = os.path.abspath(path)
            docs.extend(loaded_docs)

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


script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
vector_store = SQLiteVSS.from_texts(texts=chunks, embedding=embedding_func, table="milkyway_graph_rag", db_file="milkyway_graph_rag_vss.db", overwrite=True)


