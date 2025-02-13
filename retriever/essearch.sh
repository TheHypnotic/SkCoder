#!/bin/bash

# Query file
QUERY_FILE="../data/hearthstone/test.in"
INDEX_NAME="code-index"

# Search using embeddings
python3 searches.py "$QUERY_FILE" "$INDEX_NAME"
