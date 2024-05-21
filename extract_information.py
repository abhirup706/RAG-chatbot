import logging, pymongo
import sys
from llama_index.core import SummaryIndex,VectorStoreIndex,StorageContext
from llama_index.readers.mongodb import SimpleMongoReader
from IPython.display import Markdown, display
import os
from llama_index.core.llms import MockLLM
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
import key_param
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from flask import Flask,request,jsonify
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore


app = Flask(__name__)
os.environ['OPENAI_API_KEY'] = key_param.openai_api_key

def process_data(input_query: str,user_id: int):

    #Settings.llm = llama(model="llama2", request_timeout=60.0)
    Settings.llm = OpenAI()


    mongodb_client = pymongo.MongoClient(key_param.MONGO_URI_EMBEDDING)
    embedding_collection_name = "products_ai_embedding"

    # Instantiate the vector store
    atlas_vector_search = MongoDBAtlasVectorSearch(
        mongodb_client,
        db_name = "llamaindex_db",
        collection_name = embedding_collection_name,
        index_name = "vector_index"
    )

    #vector_store_context = StorageContext.from_defaults(vector_store=atlas_vector_search)

    index = VectorStoreIndex.from_vector_store(
    atlas_vector_search
    )

    #chat_store = SimpleChatStore()


    loaded_chat_store = SimpleChatStore.from_persist_path(
        persist_path="chat_store.json"
    )

    chat_memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000,
        chat_store=loaded_chat_store,
        chat_store_key=user_id,
    )
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=chat_memory,
        system_prompt=(
            "You are a chatbot, able to have normal interactions, as well as talk"
        ),
    )

    response = chat_engine.chat(input_query)
    loaded_chat_store.persist(persist_path="chat_store.json")


    # set Logging to DEBUG for more detailed outputs
    #llm = MockLLM()
    #query_engine = index.as_query_engine()
    #query = input_query
    #response = query_engine.query(query)
    


    #print("User : "+query)
    #print("Chatbot : "+str(response))

    #return("response:"+str(response))
    return {"response": f"{response}"}

@app.route('/process', methods=['POST'])

def process():
    data = request.json  # Get the JSON payload from the request
    query = data.get('query')  # Extract the 'query' field
    #status = data.get('status')
    user_id = data.get('userId')
    if query is None:
        return jsonify({"error": "No query provided"}), 400  # Handle the case where 'query' is not provided
    result = process_data(query,user_id)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

