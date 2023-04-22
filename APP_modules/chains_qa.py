import os
import requests
import importlib
from urllib.parse import urlparse


from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI

from flask import Flask, jsonify, request
from flask_restful import Resource, Api




def get_file_extension(url):
  
  parsed_url = urlparse(url)
  file_extension = os.path.splitext(parsed_url.path)[-1]
  return file_extension.lower()

def chains_qa_file_handler(url, namespace, query):
    # Get the file extension from the URL
    file_extension = get_file_extension(url)
        
    # Define the file extension to loader_type mapping
    extension_loader_mapping = {
        '.pdf': ('langchain.document_loaders', 'PyPDFLoader'),
        '.csv': ('langchain.document_loaders.csv_loader', 'CSVLoader'),
        '.pptx': ('langchain.document_loaders', 'UnstructuredPowerPointLoader'),
        '.docx': ('langchain.document_loaders', 'UnstructuredWordDocumentLoader'),
        '.png': ('langchain.document_loaders.image', 'UnstructuredImageLoader'),

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
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
  
    embeddings = OpenAIEmbeddings()
  
    index_name = "fine-tuner"
    namespace = namespace
    docsearch = Pinecone.from_documents(texts,
                                        embeddings,
                                        namespace=namespace,
                                        index_name=index_name)
  
    qa = RetrievalQA.from_chain_type(llm=OpenAI(),
                                     chain_type="stuff",
                                     retriever=docsearch.as_retriever())
  
    result = qa.run(query)
    return jsonify({'data': str(result)})
  









def chains_qa_web_handler(url, namespace, query):

  urls = [url]

  from langchain.document_loaders import UnstructuredURLLoader
  loader = UnstructuredURLLoader(urls)
  data = str(loader.load())
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  texts = text_splitter.split_text(str(data))

  embeddings = OpenAIEmbeddings()

  index_name = "fine-tuner"
  namespace = namespace
  docsearch = Pinecone.from_texts(texts,
                                      embeddings,
                                      namespace=namespace,
                                      index_name=index_name)

  qa = RetrievalQA.from_chain_type(llm=OpenAI(),
                                   chain_type="stuff",
                                   retriever=docsearch.as_retriever())

  result = qa.run(query)
  return jsonify({'data': str(result)})







def chains_qa_text_handler(text, namespace, query):
  
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  texts = text_splitter.split_text(str(text))

  embeddings = OpenAIEmbeddings()

  index_name = "fine-tuner"
  namespace = namespace
  docsearch = Pinecone.from_texts(texts,
                                      embeddings,
                                      namespace=namespace,
                                      index_name=index_name)

  qa = RetrievalQA.from_chain_type(llm=OpenAI(),
                                   chain_type="stuff",
                                   retriever=docsearch.as_retriever())

  result = qa.run(query)
  return jsonify({'data': str(result)})






