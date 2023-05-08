import os
import pathlib

import openai
import pinecone
import tqdm
from defaults import ROOT_DIR
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")
pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment="us-west1-gcp-free")


def main(root_dir):
    vector_store = Pinecone(
        index=pinecone.Index("hpe-sec"),
        embedding_function=OpenAIEmbeddings().embed_query,
        text_key="text",
        namespace="hpe-sec",
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    for dirpath, dirnames, filenames in os.walk(os.path.expanduser(root_dir)):
        for file in tqdm.tqdm(filenames):
            loader = UnstructuredHTMLLoader(pathlib.Path(dirpath).joinpath(file))
            split_docs = loader.load_and_split(splitter)
            for doc in split_docs:
                doc.metadata["source"] = doc.metadata["source"].stem + ".html"
            vector_store.from_documents(
                documents=split_docs,
                embedding=OpenAIEmbeddings(),
                index_name="hpe-sec",
                namespace="hpe-sec",
            )


if __name__ == "__main__":
    main(ROOT_DIR)
