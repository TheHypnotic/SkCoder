from codebert import get_code_embedding

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
