import os
import subprocess

import openai
from defaults import GITHUB_BASE_URL, ROOT_DIR
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DeepLake

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")


def clone_repository(repo_url, local_path):
    subprocess.run(["git", "clone", repo_url, local_path])


def load_docs(root_dir, github_base_url=None, splitter=None):
    docs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.split(".")[-1] not in (
                "png",
                "jpg",
                "ico",
                "svg",
                "bin",
                "woff2",
                "woff",
                "h5",
                "pkl",
                "path",
                "index",
                "ttf",
                "gif",
                "jpeg",
                "otf",
                "zip",
                "idx",
                "pt",
                "pth",
                "meta",
                "eot",
            ):
                try:
                    path = os.path.join(dirpath, file)
                    loader = TextLoader(path, encoding="utf-8")
                    split_docs = loader.load_and_split(splitter)
                    if github_base_url is not None:
                        for doc in split_docs:
                            doc.metadata["url"] = (
                                github_base_url + path[len(root_dir) :]
                            )
                    docs.extend(split_docs)
                except Exception as e:
                    print(
                        f"Exception {e} raised for file {file} but ignored; continuing."
                    )
    return docs


def main(root_dir, folder_path="faiss_index"):
    docs = load_docs(
        root_dir,
        GITHUB_BASE_URL,
        RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0),
    )
    embeddings = OpenAIEmbeddings()

    deep_lake_path = os.environ.get("DEEPLAKE_DATASET_PATH")
    db = DeepLake(dataset_path=deep_lake_path, embedding_function=embeddings)
    db.add_documents(docs)


if __name__ == "__main__":
    main(ROOT_DIR)
