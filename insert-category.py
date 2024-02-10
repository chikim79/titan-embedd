import chromadb
from chromadb.config import Settings
import sys
import uuid
from util import *


def insert_to_vector_db(id, filename, embedding, new_category):
    chroma_client = chromadb.PersistentClient(path="chroma-db")

    image_collection = chroma_client.get_or_create_collection(
        name="some_colletion_name", metadata={"hnsw:space": "cosine"}
    )

    metadata = {"category": new_category, "filename": filename}

    image_collection.upsert(ids=[id], embeddings=[embedding], metadatas=[metadata])


print(sys.argv)

if len(sys.argv) < 3:
    print("Usage python insert-category.py [image_file_path] [category]")

file_path = sys.argv[1]
category = sys.argv[2]

file_bytes = readFileAsBase64(file_path)

body = construct_bedrock_body(file_bytes)

embedding = get_embedding_from_titan_multimodal(body)

insert_to_vector_db(str(uuid.uuid4()), file_path, embedding, category)
