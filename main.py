import os
import json
import requests

import asyncio
import httpx

from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from threading import Thread

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA


import pinecone

pinecone.init(api_key=os.environ['PINECONE_API_KEY'],
              environment=os.environ['PINECONE_ENV'])
index = pinecone.Index("fine-tuner")


app = Flask('')
api = Api(app)


@app.route('/')
def home():
  return "I'm alive"



######################################################################
##APi using Flask - CHAINS - Retrieval QA - 1
########################################################################

@app.route('/chains_qa', methods=['POST'])
def chains_qa():
  data = request.get_json()
  namespace = data['namespace']
  namespace_temp = data['namespace_temp']
  query = data['query']
  chain_type = data['chain_type']


  query = query

  
  embeddings = OpenAIEmbeddings()

  vectorstore = Pinecone(index, embeddings.embed_query, "text", namespace)
  
  results = vectorstore.similarity_search(query)
  text = [doc.page_content for doc in results]


  
  text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=0)
  texts = text_splitter.split_text(str(text))


  embeddings = OpenAIEmbeddings()
  
  index_name = "fine-tuner"                
  namespace = namespace
  docsearch = Pinecone.from_texts(texts,
                                        embeddings,
                                        namespace=namespace_temp,
                                        index_name=index_name)
  
  qa = RetrievalQA.from_chain_type(llm=OpenAI(),
                                     chain_type=chain_type,
                                     retriever=docsearch.as_retriever())
  
  result = qa.run(query)
  
  return jsonify({'answer': str(result)}, {'sources': text})





######################################################################
##APi using Flask - CHAINS - Retrieval QA - 2
########################################################################

@app.route('/chains_qa_comp', methods=['POST'])
def chains_qa_comp():
  data = request.get_json()
  namespace = data['namespace']
  namespace_temp = data['namespace_temp']
  query = data['query']
  chain_type = data['chain_type']


  from langchain.retrievers import ContextualCompressionRetriever
  from langchain.retrievers.document_compressors import LLMChainExtractor
    
  query = query

  
  
  embeddings = OpenAIEmbeddings()

  vectorstore = Pinecone(index, embeddings.embed_query, "text", namespace).as_retriever()


  llm = OpenAI(temperature=0)
  compressor = LLMChainExtractor.from_llm(llm)
  compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=vectorstore)
  
  results = compression_retriever.get_relevant_documents(query)
  text = [doc.page_content for doc in results]

  

  
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  texts = text_splitter.split_text(str(text))


  embeddings = OpenAIEmbeddings()
  
  index_name = "fine-tuner"                
  namespace = namespace
  docsearch = Pinecone.from_texts(texts,
                                        embeddings,
                                        namespace=namespace_temp,
                                        index_name=index_name)
  
  qa = RetrievalQA.from_chain_type(llm=OpenAI(),
                                     chain_type=chain_type,
                                     retriever=docsearch.as_retriever())
  
  result = qa.run(query)
  
  return jsonify({'answer': str(result)}, {'sources': text})









######################################################################
##APi using Flask - EMBEDDINGS
######################################################################



from APP_modules.embeddings import embeddings_file_handler
@app.route('/embeddings_file', methods=['POST'])
def embeddings_file():
  data = request.get_json()
  url = data['url']
  chunk_size = data['chunk_size']
  namespace = data['namespace']

  result = embeddings_file_handler(url, chunk_size, namespace)
  return (result)


from APP_modules.embeddings import embeddings_web_handler
@app.route('/embeddings_web', methods=['POST'])
def embeddings_web():
  data = request.get_json()
  url = data['url']
  chunk_size = data['chunk_size']
  namespace = data['namespace']

  result = embeddings_web_handler(url, chunk_size, namespace)
  return (result)



from APP_modules.embeddings import embeddings_text_handler
@app.route('/embeddings_text', methods=['POST'])
def embeddings_text():
  data = request.get_json()
  text = data['text']
  chunk_size = data['chunk_size']
  namespace = data['namespace']

  result = embeddings_text_handler(text, chunk_size, namespace)
  return (result)









######################################################################
##APi using Flask - AGENTS - self-ask-with-search
######################################################################

@app.route('/agents_saws', methods=['POST'])
def agents_saws():
  data = request.get_json()
  query = data['query']

  from langchain.utilities import GoogleSerperAPIWrapper
  from langchain.agents import initialize_agent, Tool
  from langchain.agents import AgentType
  
  llm = OpenAI(temperature=0)
  search = GoogleSerperAPIWrapper()
  tools = [
      Tool(
          name="Intermediate Answer",
          func=search.run,
          description="useful for when you need to ask with search"
      )
  ]
  
  self_ask_with_search = initialize_agent(tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)
  result = self_ask_with_search.run(query)

  return jsonify({'data': str(result)})




######################################################################
##APi using Flask - zero shot agent
######################################################################



import sys
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType

from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType


@app.route('/agents_zsrd', methods=['POST'])
def agents_zsrd():
  data = request.get_json()
  query = data['query']
  tools_string = data['tools']

  llm = OpenAI(temperature=0.0)
  math_llm = OpenAI(temperature=0.0)
  tools = load_tools(
      tools_string, 
      llm=llm,
  )
  
  agent_chain = initialize_agent(
      tools,
      llm,
      agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
      verbose=True,
  )
  
  result = agent_chain.run(query)
  
  return jsonify({'data': str(result)})







######################################################################
##APi using Flask - BabyAGI
######################################################################


@app.route('/agents_babyagi', methods=['POST'])
def agents_babyagi():
  data = request.get_json()
  objective = data['objective']
  thread_item = data['thread_item']
  iterations = data['iterations']

  
  from collections import deque
  from typing import Dict, List, Optional, Any
  
  from langchain import LLMChain, OpenAI, PromptTemplate
  from langchain.llms import BaseLLM
  from langchain.vectorstores.base import VectorStore
  from pydantic import BaseModel, Field
  from langchain.chains.base import Chain
  
  
  from langchain.vectorstores import FAISS
  from langchain.docstore import InMemoryDocstore
  
  # Define your embedding model
  embeddings_model = OpenAIEmbeddings()
  
  # Initialize the vectorstore as empty
  import faiss
  embedding_size = 1536
  index = faiss.IndexFlatL2(embedding_size)
  vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
  
    
  
  # Define the Chains
    
  class TaskCreationChain(LLMChain):
      """Chain to generates tasks."""
  
      @classmethod
      def from_llm(cls, llm: BaseLLM, verbose: bool = False) -> LLMChain:
          """Get the response parser."""
          task_creation_template = (
              "You are an task creation AI that uses the result of an execution agent"
              " to create new tasks with the following objective: {objective},"
              " The last completed task has the result: {result}."
              " This result was based on this task description: {task_description}."
              " These are incomplete tasks: {incomplete_tasks}."
              " Based on the result, create new tasks to be completed"
              " by the AI system that do not overlap with incomplete tasks."
              " Return the tasks as an array."
          )
          prompt = PromptTemplate(
              template=task_creation_template,
              input_variables=["result", "task_description", "incomplete_tasks", "objective"],
          )
          return cls(prompt=prompt, llm=llm, verbose=verbose)
  
  
  class TaskPrioritizationChain(LLMChain):
      """Chain to prioritize tasks."""
  
      @classmethod
      def from_llm(cls, llm: BaseLLM, verbose: bool = False) -> LLMChain:
          """Get the response parser."""
          task_prioritization_template = (
              "You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing"
              " the following tasks: {task_names}."
              " Consider the ultimate objective of your team: {objective}."
              " Do not remove any tasks. Return the result as a numbered list, like:"
              " #. First task"
              " #. Second task"
              " Start the task list with number {next_task_id}."
          )
          prompt = PromptTemplate(
              template=task_prioritization_template,
              input_variables=["task_names", "next_task_id", "objective"],
          )
          return cls(prompt=prompt, llm=llm, verbose=verbose)
  
  
  
  class ExecutionChain(LLMChain):
      """Chain to execute tasks."""
  
      @classmethod
      def from_llm(cls, llm: BaseLLM, verbose: bool = False) -> LLMChain:
          """Get the response parser."""
          execution_template = (
              "You are an AI who performs one task based on the following objective: {objective}."
              " Take into account these previously completed tasks: {context}."
              " Your task: {task}."
              " Response:"
          )
          prompt = PromptTemplate(
              template=execution_template,
              input_variables=["objective", "context", "task"],
          )
          return cls(prompt=prompt, llm=llm, verbose=verbose)
  
   
  
  # Define the BabyAGI Controller
    
  
  def get_next_task(task_creation_chain: LLMChain, result: Dict, task_description: str, task_list: List[str], objective: str) -> List[Dict]:
      """Get the next task."""
      incomplete_tasks = ", ".join(task_list)
      response = task_creation_chain.run(result=result, task_description=task_description, incomplete_tasks=incomplete_tasks, objective=objective)
      new_tasks = response.split('\n')
      return [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]
  
  
  
  
  def prioritize_tasks(task_prioritization_chain: LLMChain, this_task_id: int, task_list: List[Dict], objective: str) -> List[Dict]:
      """Prioritize tasks."""
      task_names = [t["task_name"] for t in task_list]
      next_task_id = int(this_task_id) + 1
      response = task_prioritization_chain.run(task_names=task_names, next_task_id=next_task_id, objective=objective)
      new_tasks = response.split('\n')
      prioritized_task_list = []
      for task_string in new_tasks:
          if not task_string.strip():
              continue
          task_parts = task_string.strip().split(".", 1)
          if len(task_parts) == 2:
              task_id = task_parts[0].strip()
              task_name = task_parts[1].strip()
              prioritized_task_list.append({"task_id": task_id, "task_name": task_name})
      return prioritized_task_list
  
  
  
  
  def _get_top_tasks(vectorstore, query: str, k: int) -> List[str]:
      """Get the top k tasks based on the query."""
      results = vectorstore.similarity_search_with_score(query, k=k)
      if not results:
          return []
      sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))
      return [str(item.metadata['task']) for item in sorted_results]
  
  all_results = []
  def execute_task(vectorstore, execution_chain: LLMChain, objective: str, task: str, thread_item: str, k: int = 5) -> str:
      """Execute a task."""
      context = _get_top_tasks(vectorstore, query=objective, k=k)
      result = execution_chain.run(objective=objective, context=context, task=task)
      all_results.append(result)  # Append the result to the all_results list
  
      # Send the result to the specific endpoint
      result_endpoint = "https://ai-finetune.bubbleapps.io/version-live/api/1.1/wf/replittest"
      result_data = {"result": result, "thread_item": thread_item, "iterations": iterations}
      requests.post(result_endpoint, json=result_data)
  
      return result
  

  
  
  class BabyAGI(Chain, BaseModel):
      """Controller model for the BabyAGI agent."""
  
      task_list: deque = Field(default_factory=deque)
      task_creation_chain: TaskCreationChain = Field(...)
      task_prioritization_chain: TaskPrioritizationChain = Field(...)
      execution_chain: ExecutionChain = Field(...)
      task_id_counter: int = Field(1)
      vectorstore: VectorStore = Field(init=False)
      max_iterations: Optional[int] = None
          
      class Config:
          """Configuration for this pydantic object."""
          arbitrary_types_allowed = True
  
      def add_task(self, task: Dict):
          self.task_list.append(task)
  
      def print_task_list(self):
          print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
          for t in self.task_list:
              print(str(t["task_id"]) + ": " + t["task_name"])
  
      def print_next_task(self, task: Dict):
          print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
          print(str(task["task_id"]) + ": " + task["task_name"])
  
      def print_task_result(self, result: str):
          print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
          print(result)
                  
      @property
      def input_keys(self) -> List[str]:
          return ["objective"]
      
      @property
      def output_keys(self) -> List[str]:
          return []
  
      def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
          """Run the agent."""
          objective = inputs['objective']
          first_task = inputs.get("first_task", "Make a todo list")
          self.add_task({"task_id": 1, "task_name": first_task})
          num_iters = 0
          while True:
              if self.task_list:
                  self.print_task_list()
  
                  # Step 1: Pull the first task
                  task = self.task_list.popleft()
                  self.print_next_task(task)
  
                  # Step 2: Execute the task
                  result = execute_task(
                      self.vectorstore, self.execution_chain, objective, task["task_name"], thread_item
)
                  this_task_id = int(task["task_id"])
                  self.print_task_result(result)
  
                  # Step 3: Store the result in Pinecone
                  result_id = f"result_{task['task_id']}"
                  self.vectorstore.add_texts(
                      texts=[result],
                      metadatas=[{"task": task["task_name"]}],
                      ids=[result_id],
                  )
  
                  # Step 4: Create new tasks and reprioritize task list
                  new_tasks = get_next_task(
                      self.task_creation_chain, result, task["task_name"], [t["task_name"] for t in self.task_list], objective
                  )
                  for new_task in new_tasks:
                      self.task_id_counter += 1
                      new_task.update({"task_id": self.task_id_counter})
                      self.add_task(new_task)
                  self.task_list = deque(
                      prioritize_tasks(
                          self.task_prioritization_chain, this_task_id, list(self.task_list), objective
                      )
                  )
              num_iters += 1
              if self.max_iterations is not None and num_iters == self.max_iterations:
                  print("\033[91m\033[1m" + "\n*****TASK ENDING*****\n" + "\033[0m\033[0m")
                  break
          return {}
  
      @classmethod
      def from_llm(
          cls,
          llm: BaseLLM,
          vectorstore: VectorStore,
          verbose: bool = False,
          **kwargs
      ) -> "BabyAGI":
          """Initialize the BabyAGI Controller."""
          task_creation_chain = TaskCreationChain.from_llm(
              llm, verbose=verbose
          )
          task_prioritization_chain = TaskPrioritizationChain.from_llm(
              llm, verbose=verbose
          )
          execution_chain = ExecutionChain.from_llm(llm, verbose=verbose)
          return cls(
              task_creation_chain=task_creation_chain,
              task_prioritization_chain=task_prioritization_chain,
              execution_chain=execution_chain,
              vectorstore=vectorstore,
              **kwargs
          )
  
  
  
  # Run the BabyAGI
  
  OBJECTIVE = objective
  
  
  llm = OpenAI(temperature=0)
  
  # Logging of LLMChains
  verbose=False
  # If None, will keep on going forever
  max_iterations: Optional[int] = iterations
  baby_agi = BabyAGI.from_llm(
      llm=llm,
      vectorstore=vectorstore,
      verbose=verbose,
      max_iterations=max_iterations
  )
  
  results = baby_agi({"objective": OBJECTIVE})


  # Return all_results as a JSON object
  return jsonify({"all_results": all_results, "thread_item": thread_item, "iterations": iterations})









######################################################################
##APi using Flask - AutoGPT
######################################################################

@app.route('/agents_autogpt', methods=['POST'])
def agents_autogpt():
  data = request.get_json()
  objective = data['objective']
  thread_item = data['thread_item']
  iterations = data['iterations']



  
  from langchain.utilities import GoogleSerperAPIWrapper
  from langchain.agents import Tool
  from langchain.tools.file_management.write import WriteFileTool
  from langchain.tools.file_management.read import ReadFileTool
  from langchain.tools.human.tool import HumanInputRun
  
  
  search = GoogleSerperAPIWrapper()
  tools = [
      Tool(
          name = "search",
          func=search.run,
          description="useful for when you need to answer questions about current events. You should ask targeted questions"
      ),
      WriteFileTool(),
      ReadFileTool(),
  ]
  
  
  from langchain.vectorstores import FAISS
  from langchain.docstore import InMemoryDocstore
  from langchain.embeddings import OpenAIEmbeddings
  
  
  # Define your embedding model
  embeddings_model = OpenAIEmbeddings()
  # Initialize the vectorstore as empty
  import faiss
  
  embedding_size = 1536
  index = faiss.IndexFlatL2(embedding_size)
  vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
  
  
  from langchain.experimental import AutoGPT
  from langchain.chat_models import ChatOpenAI
  
  
  agent = AutoGPT.from_llm_and_tools(
      ai_name="Tom",
      ai_role="Assistant",
      tools=tools,
      llm=ChatOpenAI(temperature=0),
      memory=vectorstore.as_retriever()
  )
  
  # Set verbose to be true
  agent.chain.verbose = True
  

  results = asyncio.run(agent.run([objective], thread_item, max_iterations=iterations))








######################################################################
##RUN THE APP
######################################################################

def run():
  app.run(host='0.0.0.0', port=7228)


t = Thread(target=run)
t.start()

