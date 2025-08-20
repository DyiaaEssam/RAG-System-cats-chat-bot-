from flask import Flask, request, render_template_string, jsonify
import ollama

# Load the dataset
df = []
with open('cat-facts.txt', 'r') as file:
    dataset = file.readlines()
    print(f'Loaded {len(df)} entries')

# Models
EMBEDDING_MODEL = "nomic-embed-text"  
LANGUAGE_MODEL = "llama3"
# Vector DB in memory
VECTOR_DB = []

def add_chunk_to_database(chunk):
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))

for i, chunk in enumerate(dataset):
    add_chunk_to_database(chunk)
    print(f'Added chunk {i+1}/{len(dataset)} to the database')


import numpy as np
def cosine_similarity(a,b):
    a,b=np.array(a),np.array(b)
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))  

def retrieve(query, top_n=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Flask app
app = Flask(__name__)

chat_history = []


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        query = request.form.get("query")

        # 🔥 هنا بندمج RAG system
        retrieved_knowledge = retrieve(query)
        instruction_prompt = f"""You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{chr(10).join([f' - {chunk}' for chunk, _ in retrieved_knowledge])}
"""
        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {"role": "system", "content": instruction_prompt},
                {"role": "user", "content": query},
            ],
            stream=True,
        )
        answer = "".join([chunk["message"]["content"] for chunk in stream])

        chat_history.append({"question": query, "answer": answer})

    

    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>🐱 Cat Facts RAG Chatbot</title>
  <style>
    body {
      font-family: "Segoe UI", Arial, sans-serif;
      background: #f0f2f5;
      margin: 0; padding: 0;
      display: flex;
      justify-content: center;
      align-items: flex-start;
      min-height: 100vh;
    }
    .container {
      background: white;
      margin-top: 40px;
      width: 600px;
      border-radius: 12px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
      padding: 25px;
    }
    h2 {
      text-align: center;
      color: #333;
      margin-top: 0;
    }
    .chat-box {
      max-height: 400px;
      overflow-y: auto;
      border: 1px solid #eee;
      padding: 15px;
      border-radius: 8px;
      margin-bottom: 20px;
      background: #fafafa;
    }
    .message {
      margin-bottom: 15px;
    }
    .question {
      background: #e1f5fe;
      padding: 10px;
      border-radius: 8px;
      margin-bottom: 5px;
    }
    .answer {
      background: #f1f8e9;
      padding: 10px;
      border-radius: 8px;
    }
    textarea {
      width: 100%;
      height: 70px;
      resize: none;
      padding: 10px;
      font-size: 15px;
      border: 1px solid #ccc;
      border-radius: 8px;
      margin-top: 5px;
    }
    button {
      margin-top: 10px;
      width: 100%;
      padding: 12px;
      font-size: 16px;
      background: #4CAF50;
      color: white;
      border: none;
      border-radius: 8px;
      cursor: pointer;
    }
    button:hover {
      background: #45a049;
    }
  </style>
</head>
<body>
  <div class="container">
    <h2>🐱 Cat Facts RAG Chatbot</h2>
    <div class="chat-box">
      {% for item in chat_history %}
        <div class="message">
          <div class="question"><strong>You:</strong> {{ item.question }}</div>
          <div class="answer"><strong>Bot:</strong> {{ item.answer }}</div>
        </div>
      {% endfor %}
    </div>
    <form method="post">
      <label for="query"><strong>Ask something:</strong></label>
      <textarea id="query" name="query" placeholder="Type your question..."></textarea>
      <button type="submit">Send</button>
    </form>
  </div>
</body>
</html>
""", chat_history=chat_history)



@app.route("/ask", methods=["POST"])
def ask_api():
    data = request.json
    query = data.get("query")
    retrieved_knowledge = retrieve(query)
    instruction_prompt = f"""You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{chr(10).join([f' - {chunk}' for chunk, _ in retrieved_knowledge])}
"""
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {"role": "system", "content": instruction_prompt},
            {"role": "user", "content": query},
        ],
        stream=True,
    )
    answer = "".join([chunk["message"]["content"] for chunk in stream])
    return jsonify({"answer": answer, "context": [c for c, _ in retrieved_knowledge]})

if __name__ == "__main__":
    app.run(debug=True)
