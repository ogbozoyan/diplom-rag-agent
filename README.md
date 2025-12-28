```bash
docker run --name pgvector-container \
  -e POSTGRES_USER=langchain \
  -e POSTGRES_PASSWORD=langchain \
  -e POSTGRES_DB=langchain \
  -p 6024:5432 \
  -d pgvector/pgvector:pg16
```

```bash
export DOCS_DIR="/Users/onbozoyan/PycharmProjects/agent-rag-diplom/data/docs"
export PGVECTOR_URL="postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
export LOG_LEVEL=INFO

# важно: чтобы не делать probe на OpenAI (и не ловить лишний API call)
export VECTOR_SIZE=1536   # OpenAI text-embedding-3-small
# export VECTOR_SIZE=384  # sentence-transformers/all-MiniLM-L6-v2

# если хочешь пересобрать таблицы
export REINDEX=false

export LANGSMITH_TRACING=true
export LANGSMITH_ENDPOINT=https://api.smith.langchain.com
export LANGSMITH_API_KEY=<LANGSMITH_API_KEY>
export LANGSMITH_PROJECT=default
export OPENAI_API_KEY=<OPENAI_API_KEY>
```