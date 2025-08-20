

---

### 💡 What the project does

* Loads a dataset (in my case: `cat-facts.txt` with 150 fun cat facts 🐱).
* Generates **embeddings** for each text chunk using Ollama (`nomic-embed-text`).
* Stores these embeddings in an in-memory vector database.
* When a user asks a question → it embeds the query → retrieves the most relevant chunks using **cosine similarity** → sends them to a language model (`llama3`) to generate a grounded answer.

---

### ⚙️ The workflow

1. **Data Loading** → read and split the dataset.
2. **Embedding & Storage** → convert chunks into embeddings with Ollama.
3. **Retriever** → `retrieve(query)` finds the top-N relevant chunks.
4. **Flask Web App** → a simple web UI where you type your question and get an answer.
5. **Chat History** → every Q\&A is stored and displayed like a chat conversation.

---

### 🖥️ The result

A small web chatbot where you can ask things like:
👉 *“Tell me something interesting about cats”*
and it responds using the actual dataset rather than hallucinating.



