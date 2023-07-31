import os
import queue
import threading

import openai
import pinecone
import streamlit as st
import tiktoken
from defaults import MODEL
from dotenv import load_dotenv
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.vectorstores import Pinecone

# Load environment variables from a .env file (containing OPENAI_API_KEY)
load_dotenv()
# Set the OpenAI API key from the environment variable
try:
    openai.api_key = os.environ.get("OPENAI_API_KEY")
except KeyError:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
try:
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"], environment="us-west1-gcp-free"
    )
except KeyError:
    pinecone.init(
        api_key=st.secrets["PINECONE_API_KEY"], environment="us-west1-gcp-free"
    )
DB = Pinecone(
    index=pinecone.Index("hpe-sec-index"),
    embedding_function=OpenAIEmbeddings().embed_query,
    text_key="text",
    namespace="hpe-sec-index",
)

encoding = tiktoken.encoding_for_model(MODEL)


def get_system_prompt_with_context(context: str) -> str:
    system_prompt_template = """Given the following context, answer the following question about Hewlett Packard Enterprise's (HPE) filings with the Securities Exchange Commission (SEC). Try to be as detailed as possible and reference the components that you are looking at. Keep in mind that these are selected passages from larger documents and additional passages may be added during the conversation.
        Only reference the passages that you have been provided with. If you do not know the answer, say "I don't know"; do not make up information.

        Now, here is the relevant context from various passages:

        Context: {context}

        A chat history with the user's most recent question at the end now follows.
        """
    return system_prompt_template.format(context=context)


def get_context_from_prompt(prompt: str, k: int = 4) -> str:
    docs = DB.similarity_search(prompt, k)

    context = "\n\n".join(
        [f'From file {d.metadata["source"]}:\n' + str(d.page_content) for d in docs]
    )

    return context


def get_full_query_from_chat_history():
    token_limit = 4000
    latest_prompt = st.session_state.human[-1]
    context = get_context_from_prompt(latest_prompt)
    st.session_state.context.append(context)
    system_message = get_system_prompt_with_context(" ".join(st.session_state.context))
    latest_prompt_tokens = len(encoding.encode(latest_prompt))
    system_message_tokens = len(encoding.encode(system_message))
    token_limit -= latest_prompt_tokens + system_message_tokens
    reversed_other_msgs = []
    # See Sec. 6 of https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktokn.ipynb
    # for token counting rules.
    chat_history = [
        None
        for _ in range(len(st.session_state.ai[1:]) + len(st.session_state.human[:-1]))
    ]
    chat_history[::2] = st.session_state.ai[1:]
    if len(chat_history):
        chat_history[1::2] = st.session_state.human[:-1]
    for idx, msg in enumerate(reversed(chat_history)):
        # The ai message will always be last
        msg_type = HumanMessage if idx % 2 else AIMessage
        token_limit -= len(encoding.encode(msg)) + 4
        if token_limit >= 0:
            reversed_other_msgs.append(msg_type(content=msg))
    full_query = (
        [SystemMessage(content=system_message)]
        + list(reversed(reversed_other_msgs))
        + [HumanMessage(content=latest_prompt)]
    )
    return full_query


# For streaming responses. From https://github.com/mtenenholtz/chat-twitter/blob/main/backend/main.py
class ThreadedGenerator:
    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is StopIteration:
            raise item
        return item

    def send(self, data):
        self.queue.put(data)

    def close(self):
        self.queue.put(StopIteration)


class ChainStreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def on_llm_new_token(self, token: str, **kwargs):
        self.gen.send(token)


def llm_thread(gen, query):
    try:
        llm = ChatOpenAI(
            model_name=MODEL,
            verbose=True,
            streaming=True,
            callback_manager=CallbackManager([ChainStreamHandler(gen)]),
            temperature=0.7,
        )

        llm(query)

    finally:
        gen.close()


def get_chat_generator(query):
    gen = ThreadedGenerator()
    threading.Thread(target=llm_thread, args=(gen, query)).start()
    return gen


st.set_page_config(page_title="HPE SEC Chatbot")
st.title("HPE SEC Chatbot")

if "ai" not in st.session_state:
    st.session_state["ai"] = ["Hi! What would you like to know about HPE SEC filings?"]
if "human" not in st.session_state:
    st.session_state["human"] = []
if "context" not in st.session_state:
    st.session_state["context"] = []


st.markdown(st.session_state.ai[0])


user_input = st.text_input(label="user input", key="input", label_visibility="hidden")
if user_input:
    st.session_state.human.append(user_input)
    query = get_full_query_from_chat_history()
    resp_box = st.empty()
    st.session_state.ai.append("")
    for response in get_chat_generator(query):
        st.session_state.ai[-1] += response
        resp_box.markdown(st.session_state.ai[-1])
