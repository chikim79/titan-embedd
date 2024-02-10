import chromadb
from chromadb.config import Settings
import sys
from util import *


def find_closest(embedding):
    chroma_client = chromadb.PersistentClient(path="chroma-db")

    image_collection = chroma_client.get_or_create_collection(
        name="some_colletion_name", metadata={"hnsw:space": "cosine"}
    )

    result = image_collection.query(
        query_embeddings=[embedding],
        n_results=1,
        include=["embeddings", "metadatas", "distances"],
    )

    found_category = result["metadatas"][0][0]["category"]
    original_filename = result["metadatas"][0][0]["filename"]
    distance = result["distances"][0][0]

    print(
        f"FOUND category: {found_category} from {original_filename} with distance {distance}"
    )


print(sys.argv)

if len(sys.argv) < 2:
    print("Usage find_closest.py [image_file_path]")

file_path = sys.argv[1]

file_bytes = readFileAsBase64(file_path)

body = construct_bedrock_body(file_bytes)

embedding = get_embedding_from_titan_multimodal(body)

find_closest(embedding)
