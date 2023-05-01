import os

import openai
import streamlit as st
import tiktoken
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from streamlit_chat import message

MODEL = "gpt-3.5-turbo"

# Load environment variables from a .env file (containing OPENAI_API_KEY)
load_dotenv()
# Set the title for the Streamlit app
st.title("Chat with Determined")
# Set the OpenAI API key from the environment variable
try:
    openai.api_key = os.environ.get("OPENAI_API_KEY")
except KeyError:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
try:
    active_loop_data_set_path = os.environ.get("DEEPLAKE_DATASET_PATH")
except KeyError:
    active_loop_data_set_path = st.secrets["DEEPLAKE_DATASET_PATH"]


# Create an instance of OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# Create an instance of DeepLake with the specified dataset path and embeddings
db = DeepLake(
    dataset_path=active_loop_data_set_path,
    read_only=True,
    embedding_function=embeddings,
)


def get_text():
    # Create a Streamlit input field and return the user's input
    input_text = st.text_input("", key="input")
    return input_text


def search_db(db, query):
    # Create a retriever from the DeepLake instance
    retriever = db.as_retriever()
    # Set the search parameters for the retriever
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 100
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 10
    # Create a ChatOpenAI model instance
    model = ChatOpenAI(model=MODEL)
    # Create a RetrievlQA instance from the model and retriever
    qa = RetrievalQA.from_llm(model, retriever=retriever)
    # Return the result of the query
    return qa.run(query)


# See Sec. 6 of https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
# for token counting rules.

# Initialize the session state for generated responses and human inputs
if "generated" not in st.session_state:
    st.session_state["generated"] = ["What would you like to know about Determined AI?"]

if "human" not in st.session_state:
    st.session_state["human"] = ["Hello!"]

# Get the user's input from the text input field
user_input = get_text()

# If there is user input, search for a response using the search_db function
if user_input:
    output = search_db(db, user_input)
    st.session_state.human.append(user_input)
    st.session_state.generated.append(output)

# If there are generated responses, display the conversation using Streamlit messages
if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"])):
        message(st.session_state["human"][i], is_user=True, key=str(i) + "_user")
        message(st.session_state["generated"][i], key=str(i))
