import os
import requests
import importlib
from urllib.parse import urlparse


from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

import pinecone

pinecone.init(api_key=os.environ['PINECONE_API_KEY'],
              environment=os.environ['PINECONE_ENV'])
index = pinecone.Index("fine-tuner")
embeddings = OpenAIEmbeddings()





from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI

from flask import Flask, jsonify, request
from flask_restful import Resource, Api






def get_file_extension(url):
  
  parsed_url = urlparse(url)
  file_extension = os.path.splitext(parsed_url.path)[-1]
  return file_extension.lower()

def embeddings_file_handler(url, chunk_size, namespace):
  
    # Get the file extension from the URL
    file_extension = get_file_extension(url)
        
    # Define the file extension to loader_type mapping
    extension_loader_mapping = {
        '.pdf': ('langchain.document_loaders', 'PyPDFLoader'),
        '.csv': ('langchain.document_loaders.csv_loader', 'CSVLoader'),
        '.pptx': ('langchain.document_loaders', 'UnstructuredPowerPointLoader'),
        '.docx': ('langchain.document_loaders', 'UnstructuredWordDocumentLoader'),
        # ...
    }

    # Get the loader_type based on the file extension
    loader_type = extension_loader_mapping.get(file_extension)
    if loader_type is None:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    # Import and instantiate the appropriate loader based on loader_type
    module_name, loader_class_name = loader_type
    if not (module_name and loader_class_name):
        raise ValueError(f"Unsupported loader type: {loader_type}")
    
    loader_module = importlib.import_module(module_name)
    loader_class = getattr(loader_module, loader_class_name)

    response = requests.get(url)
    file_path = os.path.join("APP_user_data", f"{namespace}")
    with open(file_path, "wb") as f:
        f.write(response.content)

    loader = loader_class(file_path)
    documents = loader.load()
  
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap  = 20,
        length_function = len,
)
  
    texts = text_splitter.split_documents(documents)
    vectorstore = Pinecone(index, embeddings.embed_query, "text")

  
    upserted_ids = []
    for text in texts:
        text_str = str(text)
        upsert_id = vectorstore.add_texts([text_str], namespace=namespace)
        upserted_ids.extend(upsert_id)

  
    return jsonify({'data': [str(document) for document in upserted_ids]})





def embeddings_web_handler(url, chunk_size, namespace):

  urls = [url]
  
  from langchain.document_loaders import UnstructuredURLLoader
  loader = UnstructuredURLLoader(urls)
  data = str(loader.load())
  
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = chunk_size,
      chunk_overlap  = 20,
      length_function = len,
)

  texts = text_splitter.split_text(str(data))
  vectorstore = Pinecone(index, embeddings.embed_query, "text")
  
  upserted_ids = []
  for text in texts:
      upsert_id = vectorstore.add_texts([text], namespace=namespace)
      upserted_ids.extend(upsert_id)

  return jsonify({'data': (upserted_ids)})





def embeddings_text_handler(text, chunk_size, namespace):

  data = str(text)
  
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = chunk_size,
      chunk_overlap  = 20,
      length_function = len,
)

  texts = text_splitter.split_text(str(data))
  vectorstore = Pinecone(index, embeddings.embed_query, "text")
  
  upserted_ids = []
  for text in texts:
      upsert_id = vectorstore.add_texts([text], namespace=namespace)
      upserted_ids.extend(upsert_id)

  return jsonify({'data': (upserted_ids)})







