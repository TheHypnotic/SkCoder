from elasticsearch import Elasticsearch
import numpy as np
from codebert import get_code_embedding
# Initialize Elasticsearch client
es = Elasticsearch()

# Define the index with a dense_vector field
index_name = 'code-index'

if not es.indices.exists(index=index_name):
    es.indices.create(
        index=index_name,
        body={
            "mappings": {
                "properties": {
                    "code": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 768  
                    }
                }
            }
        }
    )

def index_code(code, index_name):
    embedding = get_code_embedding(code)  # Get the embedding for the code snippet
    doc = {
        "code": code,
        "embedding": embedding.tolist()  # Convert to list before storing
    }
    es.index(index=index_name, document=doc)


def search_similar_code(query_code, index_name):
    query_embedding = get_code_embedding(query_code)  

    response = es.search(
        index=index_name,
        body={
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding.tolist(),
                        "k": 10  # Return top 10 most similar results
                    }
                }
            }
        }
    )

    # Print the most similar results
    for hit in response['hits']['hits']:
        print(f"Score: {hit['_score']}, Code: {hit['_source']['code']}")

