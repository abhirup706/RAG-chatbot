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


os.environ['OPENAI_API_KEY'] = key_param.openai_api_key
#Settings.llm = llama(model="llama2", request_timeout=60.0)
#Settings.llm = OpenAI()


#logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

host = key_param.mongo_host
port = key_param.mongo_port
db_name = key_param.mongo_db_name

source_collection_name = "products_ai"
embedding_collection_name = "products_ai_embedding"

# query_dict is passed into db.collection.find()
#query_dict = {'name':'{ \"$exists\": true }'}
query_dict={}
field_names = ["name"]
metadata_names = ["code","uom"]
reader = SimpleMongoReader(host, port)
#print(db_name)
#print(collection_name)
#print(field_names)
documents = reader.load_data(
    db_name, source_collection_name,field_names,query_dict=query_dict, separator=', ', max_docs=10000, metadata_names=metadata_names
)


# Connect to your Atlas cluster
mongodb_client = pymongo.MongoClient(key_param.MONGO_URI_EMBEDDING)

# Instantiate the vector store
atlas_vector_search = MongoDBAtlasVectorSearch(
    mongodb_client,
    db_name = "llamaindex_db",
    collection_name = embedding_collection_name,
    index_name = "vector_index"
)

vector_store_context = StorageContext.from_defaults(vector_store=atlas_vector_search)

vector_store_index = VectorStoreIndex.from_documents(
   documents, storage_context=vector_store_context, show_progress=True
)
