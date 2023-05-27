from __future__ import annotations

import os
import json
import requests

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
from langchain.chat_models import ChatAnthropic

from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain




from typing import Any, Dict, List, Optional
from pydantic import Extra
from langchain.callbacks.manager import (
  AsyncCallbackManagerForChainRun,
  CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.base_language import BaseLanguageModel
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts import PromptTemplate





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
  query = data['query']
  chain_type = data['chain_type']
  temp = data['temp']
  LLM = data['LLM']



  embeddings = OpenAIEmbeddings()
  
  index_name = "fine-tuner"                
  namespace = namespace
  docsearch = Pinecone.from_existing_index(index_name,
                                        embeddings,
                                        namespace=namespace
                                          )


    # docsearch = Pinecone.from_existing_index(index_name, embeddings)
    # docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)


  if LLM == "anthropic":
     llm = ChatAnthropic(temperature=temp)
  elif LLM == "openai":
     llm = OpenAI(temperature=temp)
  elif LLM == "gpt-3.5-turbo":
     llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=temp)
  elif LLM == "gpt-4":
     llm = ChatOpenAI(model_name='gpt-4', temperature=temp)
  else:
      return jsonify({"error": "Invalid LLM value"}), 400
  
  
  qa = RetrievalQA.from_chain_type(llm,
                                     chain_type=chain_type,
                                     retriever=docsearch.as_retriever())
  
  result = qa.run(query)
  
  return jsonify({'answer': str(result)})






######################################################################
##APi using Flask - EMBEDDINGS
######################################################################



from APP_modules.embeddings import embeddings_file_handler

def process_and_notify(url, namespace, id):
    logger.info("Processing started")
    result = embeddings_file_handler(url, namespace, id)
    logger.info(f"Processing completed. Result: {result}")

    # Add the namespace to the result
    result['id'] = id

    # Convert the result to JSON
    result_json = json.dumps(result)

    # Send the result to the Bubble app
    bubble_endpoint = "https://ai-finetune.bubbleapps.io/api/1.1/wf/webhook_upsert"
    headers = {'Content-Type': 'application/json'}
    response = requests.post(bubble_endpoint, data=result_json, headers=headers)

    logger.info(f"Response from Bubble app: {response.status_code}, {response.text}")

@app.route('/embeddings_file', methods=['POST'])
def embeddings_file():
    data = request.get_json()
    url = data['url']
    namespace = data['namespace']
    id = data['id']


    # Start a new thread to process the data and notify the Bubble app when done
    t = Thread(target=process_and_notify, args=(url, namespace, id))
    t.start()

    return jsonify({"status": "Processing started"})












from APP_modules.embeddings import embeddings_web_handler

def process_and_notify_web(url, namespace, id):
    logger.info("Processing started")
    result = embeddings_web_handler(url, namespace, id)
    logger.info(f"Processing completed. Result: {result}")

    # Add the namespace to the result
    result['id'] = id

    # Convert the result to JSON
    result_json = json.dumps(result)

    # Send the result to the Bubble app
    bubble_endpoint = "https://ai-finetune.bubbleapps.io/api/1.1/wf/webhook_upsert"
    headers = {'Content-Type': 'application/json'}
    response = requests.post(bubble_endpoint, data=result_json, headers=headers)

@app.route('/embeddings_web', methods=['POST'])
def embeddings_web():
    data = request.get_json()
    url = data['url']
    namespace = data['namespace']
    id = data['id']

    # Start a new thread to process the data and send the result to the Bubble app
    t = Thread(target=process_and_notify_web, args=(url, namespace, id))
    t.start()

    return jsonify({"status": "processing_started"})









from APP_modules.embeddings import embeddings_text_handler

def process_and_notify_text(text, namespace, id):
    result = embeddings_text_handler(text, namespace, id)
    # Add the namespace to the result

    result['id'] = id

    # Convert the result to JSON
    result_json = json.dumps(result)

    # Send the result to the Bubble app
    bubble_endpoint = "https://ai-finetune.bubbleapps.io/api/1.1/wf/webhook_upsert"
    headers = {'Content-Type': 'application/json'}
    response = requests.post(bubble_endpoint, data=result_json, headers=headers)


@app.route('/embeddings_text', methods=['POST'])
def embeddings_text():
    data = request.get_json()
    text = data['text']
    namespace = data['namespace']
    id = data['id']

    t = Thread(target=process_and_notify_text, args=(text, namespace, id))
    t.start()

    return jsonify({'status': 'Processing started'})










from APP_modules.embeddings import embeddings_twitter_handler

def process_and_notify_twitter(handle, token, number_tweets, namespace, id):
    result = embeddings_twitter_handler(handle, token, number_tweets, namespace, id)
    # Add the namespace to the result

    result['id'] = id

    # Convert the result to JSON
    result_json = json.dumps(result)

    # Send the result to the Bubble app
    bubble_endpoint = "https://ai-finetune.bubbleapps.io/api/1.1/wf/webhook_upsert"
    headers = {'Content-Type': 'application/json'}
    response = requests.post(bubble_endpoint, data=result_json, headers=headers)


@app.route('/embeddings_twitter', methods=['POST'])
def embeddings_twitter():
    data = request.get_json()   
    handle = data['handle']
    namespace = data['namespace']
    token = data['token']
    number_tweets = data['number_tweets']
    id = data['id']

    t = Thread(target=process_and_notify_twitter, args=(handle, token, number_tweets, namespace, id))
    t.start()

    return jsonify({'status': 'Processing started'})
















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
from langchain.chat_models import ChatAnthropic

from langchain.tools import AIPluginTool


@app.route('/agents_zsrd', methods=['POST'])
def agents_zsrd():
  data = request.get_json()
  query = data['query']
  tools_string = data['tools']
  additional_tools = data['additional_tools']
  temp = data['temp']
  LLM = data['LLM']


  if LLM == "anthropic":
      llm = ChatAnthropic(temperature=temp)
  elif LLM == "openai":
     llm = OpenAI(temperature=temp)

  
  # Load tools specified in the request
  tools = load_tools(
      tools_string, 
      llm=llm,
  )

  
  # Check if "klarna" is in tools_string and add the Klarna AI plugin tool if it is
  if "klarna" in additional_tools:
    tool = AIPluginTool.from_plugin_url("https://www.klarna.com/.well-known/ai-plugin.json")
    tools.append(tool)

 
  agent_chain = initialize_agent(
      tools,
      llm,
      agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
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
##APi using Flask - ImageQA Chain
######################################################################


@app.route('/chains_image', methods=['POST'])
def chains_image():
  data = request.get_json()
  query = data['query']
  history = data['history']
  image_description = data['image_description']

  
  
  class MyCustomChain(Chain):
    """
      An example of a custom chain.
      """
  
    prompt: BasePromptTemplate
    """Prompt object to use."""
    llm: BaseLanguageModel
    output_key: str = "text"  #: :meta private:
  
    class Config:
      """Configuration for this pydantic object."""
  
      extra = Extra.forbid
      arbitrary_types_allowed = True
  
    @property
    def input_keys(self) -> List[str]:
      """Will be whatever keys the prompt expects.
  
          :meta private:
          """
      return self.prompt.input_variables
  
    @property
    def output_keys(self) -> List[str]:
      """Will always return text key.
  
          :meta private:
          """
      return [self.output_key]
  
    def _call(
      self,
      inputs: Dict[str, Any],
      run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
      # Your custom chain logic goes here
      # This is just an example that mimics LLMChain
      prompt_value = self.prompt.format_prompt(**inputs)
  
      # Whenever you call a language model, or another chain, you should pass
      # a callback manager to it. This allows the inner run to be tracked by
      # any callbacks that are registered on the outer run.
      # You can always obtain a callback manager for this by calling
      # `run_manager.get_child()` as shown below.
      response = self.llm.generate_prompt(
        [prompt_value],
        callbacks=run_manager.get_child() if run_manager else None)
  
      # If you want to log something about this run, you can do so by calling
      # methods on the `run_manager`, as shown below. This will trigger any
      # callbacks that are registered for that event.
      if run_manager:
        run_manager.on_text("Log something about this run")
  
      return {self.output_key: response.generations[0][0].text}
  
    async def _acall(
      self,
      inputs: Dict[str, Any],
      run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
      # Your custom chain logic goes here
      # This is just an example that mimics LLMChain
      prompt_value = self.prompt.format_prompt(**inputs)
  
      # Whenever you call a language model, or another chain, you should pass
      # a callback manager to it. This allows the inner run to be tracked by
      # any callbacks that are registered on the outer run.
      # You can always obtain a callback manager for this by calling
      # `run_manager.get_child()` as shown below.
      response = await self.llm.agenerate_prompt(
        [prompt_value],
        callbacks=run_manager.get_child() if run_manager else None)
  
      # If you want to log something about this run, you can do so by calling
      # methods on the `run_manager`, as shown below. This will trigger any
      # callbacks that are registered for that event.
      if run_manager:
        await run_manager.on_text("Log something about this run")
  
      return {self.output_key: response.generations[0][0].text}
  
    @property
    def _chain_type(self) -> str:
      return "my_custom_chain"
  
  MyCustomChain.update_forward_refs()

  from langchain.callbacks.stdout import StdOutCallbackHandler
  from langchain.chat_models.openai import ChatOpenAI
  from langchain.prompts.prompt import PromptTemplate
  
  template_with_history = """Engage in a conversation with the user based on the given image description. Use the following format:
  
  Begin! Remember to engage in the conversation based on the image description.
  
  Previous conversation history:
  {history}
  
  Image Description:
  {image_description}
  
  New question: {input}
  {agent_scratchpad}"""
  
  chain = MyCustomChain(
    prompt=PromptTemplate.from_template(template_with_history), llm=ChatOpenAI())
  
    
  result = chain.run(
    {
      'image_description': image_description,
      'input': query,
      'agent_scratchpad': '',
      'history': history
    },
    callbacks=[StdOutCallbackHandler()])
  
  print(result)
  
  return jsonify({'answer': result})





######################################################################
##APi using Flask - CSV Agent
######################################################################


from langchain.agents import create_csv_agent

from langchain.llms import OpenAI




@app.route('/agents_csv', methods=['POST'])
def agents_csv():
  data = request.get_json()
  url = data['url']
  query = data['query']
  name = data['name']


  response = requests.get(url)
  file_path = os.path.join("APP_user_data", f"{name}")
  with open(file_path, "wb") as f:
      f.write(response.content)

  
  
  agent = create_csv_agent(OpenAI(temperature=0), file_path, verbose=True)
  
  result = agent.run(query)

  return jsonify({'answer': result})










######################################################################
##RUN THE APP
######################################################################

def run():
  app.run(host='0.0.0.0', port=7228)


t = Thread(target=run)
t.start()

