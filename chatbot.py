import os
import sys

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA

from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import OpenAIEmbeddings 
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import Chroma

import constants


import warnings
warnings.filterwarnings('ignore')


os.environ["OPENAI_API_KEY"] = constants.APIKEY

app = Flask(__name__)

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

embedding_model = OpenAIEmbeddings()

def create_index(persist=False):
    if persist and os.path.exists("persist"):
        print("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        loader = DirectoryLoader("data/")
        if persist:
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"},embedding=embedding_model).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator(embedding=embedding_model).from_loaders([loader])
    return index


@app.route('/chat', methods=['POST'])
def chat(chat_history=[]):
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

  
    result = chain({"question": query, "chat_history": chat_history})
    print(result['answer'])

    chat_history.append((query, result['answer']))
    return result['answer'], chat_history # returning the chat history so that it can be passed back in the next iteration



if __name__ == "__main__":
    PERSIST = False
    embedding_model = OpenAIEmbeddings()
    index = create_index(persist=PERSIST)
    chat("What is the capital of France?")
