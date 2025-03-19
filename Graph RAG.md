# 📊 **Graph RAG with Neo4j – Hands-on Training**  

---

## 🏆 **Objectives**  
✅ Understand RAG (Retrieval-Augmented Generation) and its importance  
✅ Learn key concepts: embeddings, vectors, similarity search, cosine similarity, chunking, tokens  
✅ Understand how RAG overcomes limitations like context length and domain knowledge  
✅ Explore how Graph RAG integrates with Neo4j for better retrieval and generation  
✅ Hands-on examples using Python, Neo4j, and OpenAI  
✅ Expand Graph RAG to handle multi-modal data (images, videos)  

---

## 🌍 **1. What is RAG (Retrieval-Augmented Generation)?**  

### 🔎 **Definition**  
RAG is an AI framework that combines **information retrieval** and **language generation** to produce more accurate and context-aware responses.  

### 🚀 **How RAG Works:**  
1. **Input:** User provides a query  
2. **Retrieval:** Relevant data is fetched from a knowledge base using embeddings and similarity search  
3. **Augmentation:** Retrieved data is added to the prompt  
4. **Generation:** LLM generates the final answer based on the augmented prompt  

---

### ✅ **Why RAG is Powerful**  
| Problem | How RAG Solves It |
|---------|--------------------|
| **Context Length Limit** | External knowledge base extends the available context |
| **Hallucination** | Grounding responses in real data reduces hallucination |
| **Domain Knowledge** | Embedding search helps include domain-specific information |
| **Real-Time Information** | Retrieval ensures the latest data is used |

---

### 🏆 **Example:**  
**Without RAG:**  
👉 *What is the capital of France?*  
→ "The capital of France is Paris."  

**With RAG:**  
👉 *What is the latest news about France's elections?*  
→ RAG retrieves recent news and provides an updated answer.  

---

## 🧠 **2. Key Concepts**  

### 📌 **Embeddings**  
- Mathematical representation of text or data in a multi-dimensional space.  
- Similar texts will have embeddings closer to each other.  
- Generated using models like **OpenAI, SentenceTransformers, HuggingFace**  

**Example:**  
```python
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
text = "France is a country in Europe."
vector = embeddings.embed_query(text)
print(vector)  # Output: [0.12, -0.34, 0.87, ...]
```

---

### 📌 **Vectors**  
- Numerical arrays representing embeddings.  
- Used to measure similarity between data points.  

**Example:**  
👉 `"Paris is the capital of France"` → `[0.12, -0.45, 0.98, -0.21, ...]`  
👉 `"Berlin is the capital of Germany"` → `[0.15, -0.43, 0.91, -0.20, ...]`  

---

### 📌 **Similarity Search**  
- Process of finding the most relevant embeddings based on a similarity score.  
- Techniques:  
  - **Cosine Similarity**  
  - **Euclidean Distance**  

---

### 📌 **Cosine Similarity**  
Measures the angle between two vectors.  
\[
\text{similarity}(A, B) = \frac{A \cdot B}{||A|| \times ||B||}
\]

**Example:**  
```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

vec1 = np.array([0.12, -0.45, 0.98])
vec2 = np.array([0.15, -0.43, 0.91])

similarity = cosine_similarity([vec1], [vec2])
print(similarity)  # Output: [[0.998]]
```

---

### 📌 **Chunking**  
- Splitting large text into smaller parts for better embedding and retrieval.  

**Example:**  
👉 "Neo4j is a graph database..." → `[Chunk 1]`, `[Chunk 2]`  

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text = "Neo4j is a graph database that stores data as nodes and edges..."
splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
chunks = splitter.split_text(text)
print(chunks)
```

---

### 📌 **Tokens**  
- Smallest unit of text processed by an LLM.  
- Example: `"Neo4j is powerful"` → `["Neo4j", "is", "powerful"]`  

**Example:**  
```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokens = tokenizer.encode("Neo4j is a graph database.")
print(tokens)  # Output: [6342, 318, 257, 20776, 5875, 13]
```

---

## 📚 **3. How Graph RAG Works with Neo4j**  

### ✅ **Graph Structure in Neo4j**  
- **Nodes** → Entities (documents, authors, keywords)  
- **Relationships** → Contextual connections between nodes  

### ✅ **Example Graph Structure:**  
**Document** → `CONNECTED_TO` → **Keyword**  
**Document** → `MENTIONS` → **Entity**  

---

### **Step 1: Setup Neo4j Connection**  
```python
from neo4j import GraphDatabase

class GraphHandler:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()
```

---

### **Step 2: Insert Data into Neo4j**  
```python
def create_node(tx, label, properties):
    query = f"CREATE (n:{label} {{ {', '.join(f'{k}: ${k}' for k in properties)} }})"
    tx.run(query, **properties)

handler = GraphHandler("bolt://localhost:7687", "neo4j", "password")
with handler.driver.session() as session:
    session.write_transaction(create_node, "Document", {
        "text": "Neo4j is a graph database...",
        "embedding": vector
    })
```

---

### **Step 3: Perform Similarity Search**  
```python
def search_similar_docs(tx, query_vector):
    query = """
    MATCH (d:Document)
    RETURN d.text, cosineSimilarity(d.embedding, $query_vector) AS score
    ORDER BY score DESC
    LIMIT 3
    """
    result = tx.run(query, query_vector=query_vector)
    return [record["d.text"] for record in result]

with handler.driver.session() as session:
    result = session.read_transaction(search_similar_docs, vector)
    print(result)
```

---

### **Step 4: Generate Response with LLM**  
```python
from langchain.llms import OpenAI

llm = OpenAI()
prompt = f"Using the following context:\n{result}\nExplain Neo4j."
response = llm(prompt)
print(response)
```

---

## 🌟 **4. Why Graph RAG is Better**  
| Problem | Solution in Graph RAG |
|---------|-----------------------|
| Context Length | Extends retrieval beyond token limits |
| Data Relationships | Graph-based connections improve contextual relevance |
| Domain Knowledge | Custom embeddings for specialized data |
| Real-Time Knowledge | Fast retrieval of the latest information |

---

## 🎯 **5. Multi-Modal Graph RAG**  

### ✅ **Graph RAG for Images**  
- Store image embeddings in Neo4j  
- Use similarity search for retrieval  
- Example: Search for a related product based on an image  

---

### ✅ **Graph RAG for Videos**  
- Store video embeddings (frame-based)  
- Connect video segments using graph relationships  
- Example: Find related tutorial videos  

---

## 🚀 **6. Best Practices**  
✅ Clean and pre-process data before embedding  
✅ Use chunking to optimize search size  
✅ Store embeddings in vector format  
✅ Create meaningful relationships in Neo4j  

---

## 🏁 **7. Summary**  
- RAG = Retrieval + Augmentation + Generation  
- Graph-based RAG improves context and relevance  
- Neo4j enhances relationship-based retrieval  
- Extend Graph RAG to multi-modal data  

---

## ✅ **8. Next Steps**  
🔹 Introduce LangGraph for workflow automation  
🔹 Fine-tune embeddings for better search  
🔹 Handle real-time updates in the graph  

---

## 🙌 **Q&A**  
