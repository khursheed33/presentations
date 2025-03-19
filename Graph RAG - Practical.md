# Practical on Graph RAG

### **Project Structure**  
```
graph_rag_project/
├── .env
├── main.py
├── config.py
├── graph_handler.py
├── embedding_handler.py
├── retrieval.py
├── generate_response.py
├── requirements.txt
└── README.md
```

---

## 1. **requirements.txt**  
Install necessary dependencies:  
```plaintext
neo4j
langchain
openai
python-dotenv
numpy
```

---

## 2. **.env**  
Store environment variables securely:  
```plaintext
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
OPENAI_API_KEY=your-openai-key
```

---

## 3. **config.py**  
Load environment variables:  
```python
import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

---

## 4. **graph_handler.py**  
Class to handle Neo4j connection and operations:  

```python
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

class GraphHandler:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def create_node(self, label, properties):
        with self.driver.session() as session:
            query = f"""
            CREATE (n:{label} {{ {', '.join(f'{k}: ${k}' for k in properties)} }})
            """
            session.run(query, **properties)

    def search_similar_docs(self, query_vector):
        with self.driver.session() as session:
            query = """
            MATCH (d:Document)
            RETURN d.text, cosineSimilarity(d.embedding, $query_vector) AS score
            ORDER BY score DESC
            LIMIT 3
            """
            result = session.run(query, query_vector=query_vector)
            return [record["d.text"] for record in result]
```

---

## 5. **embedding_handler.py**  
Class to handle embeddings:  

```python
from langchain.embeddings import OpenAIEmbeddings

class EmbeddingHandler:
    def __init__(self):
        self.embedder = OpenAIEmbeddings()

    def get_embedding(self, text):
        return self.embedder.embed_query(text)
```

---

## 6. **retrieval.py**  
Class to perform document retrieval using similarity search:  

```python
from graph_handler import GraphHandler
from embedding_handler import EmbeddingHandler

class Retrieval:
    def __init__(self):
        self.graph = GraphHandler()
        self.embedder = EmbeddingHandler()

    def store_document(self, text):
        vector = self.embedder.get_embedding(text)
        self.graph.create_node("Document", {"text": text, "embedding": vector})

    def retrieve_similar_docs(self, query):
        query_vector = self.embedder.get_embedding(query)
        results = self.graph.search_similar_docs(query_vector)
        return results
```

---

## 7. **generate_response.py**  
Class to generate LLM response based on retrieved context:  

```python
from langchain.llms import OpenAI
from retrieval import Retrieval

class GenerateResponse:
    def __init__(self):
        self.llm = OpenAI()
        self.retriever = Retrieval()

    def get_response(self, query):
        context = self.retriever.retrieve_similar_docs(query)
        prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
        response = self.llm(prompt)
        return response
```

---

## 8. **main.py**  
Main entry point to run the project:  

```python
from retrieval import Retrieval
from generate_response import GenerateResponse

def main():
    retriever = Retrieval()
    generator = GenerateResponse()

    # Step 1: Insert document into Neo4j
    retriever.store_document("Neo4j is a graph database that stores data as nodes and edges.")

    # Step 2: Retrieve and Generate Response
    query = "What is Neo4j?"
    response = generator.get_response(query)

    print("Generated Response:")
    print(response)

if __name__ == "__main__":
    main()
```

---

## 9. **README.md**  
Explain how to run the project:  

```markdown
# Graph RAG with Neo4j and OpenAI

## 📌 Setup
1. Clone the repository:
```bash
git clone https://github.com/khursheed33/graph_rag_project.git
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file:
```plaintext
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
OPENAI_API_KEY=your-openai-key
```

5. Run the project:
```bash
python main.py
```

## 🚀 How It Works
1. **Embeddings:** Convert text into vector embeddings using OpenAI.  
2. **Graph Storage:** Store embeddings and text in Neo4j.  
3. **Similarity Search:** Use cosine similarity to retrieve related documents.  
4. **LLM Generation:** Use the retrieved context to generate accurate answers.  

## ✅ Example Output:
```
Generated Response:
Neo4j is a graph database that stores data as nodes and edges, allowing for fast and connected data retrieval.
```

## 🏆 Why Graph RAG?
✅ Fast and accurate retrieval  
✅ Handles large-scale connected data  
✅ Better context handling through graph structure  
✅ Real-time knowledge augmentation  
```
