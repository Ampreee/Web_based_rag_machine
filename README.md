# **Web based Rag machine**

Web based Rag machine is made by me which two API for which First is ingest which takes url and user name and immediately return job id and job queue index.
on backend it calls a worker which then fetch pages ,extract ,chunk ,create embeddings and store.The other API we have is query which recieve query and create 
embedding of it.It does semantic similarity and fetches top k chunks Then it passes to groq api where we use LLM to summarize the chunks and return answer.

# **Setting up**

create virtual env and install requriements using this

    pip install -r requirements.txt

then create a env as same as .env.example

# **Launching**

to start backend do 

    uvicorn app.main:app --reload

to start worker 

    arq app.workers.tasks.WorkerSettings

# **Curls**

Ingest curl

    curl -X POST http://localhost:8000/ingest/ -H "accept: application/json" -H "Content-Type: application/json" -d "{\"urls\": [\"https://www.thehindu.com/news/national/gujarat/new-gujarat-cabinet-sworn-in-harsh-sanghvi-named-deputy-cm-rivaba-jadeja-among-new-faces/article70174743.ece\"], \"submitted_by\": \"Aman\"}"
Query curl

    curl -X POST http://localhost:8000/query/ -H "accept: application/json" -H "Content-Type: application/json" -d "{\"q\": \"gujarat cabinet sworn in harsh sanghvi\"}"

<img width="1447" height="356" alt="image" src="https://github.com/user-attachments/assets/a04361fb-9abe-43cb-b327-85aeaa3c4453" />

    
