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

## 🌍 **What is RAG (Retrieval-Augmented Generation)?**  

### 🔎 **Definition**  
- RAG is an AI framework that combines **information retrieval** and **language generation** to produce more accurate and context-aware responses.  
- It works by retrieving relevant information from a knowledge base and injecting it into the LLM's prompt to enhance response quality.  

---

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

## 🧠 **Key Concepts**  

### 📌 **Embeddings**  
- Mathematical representation of text or data in a multi-dimensional space.  
- Similar texts will have embeddings closer to each other.  

**Example:**  
```python
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
text = "France is a country in Europe."
vector = embeddings.embed_query(text)
print(vector)  # Output: [0.12, -0.34, 0.87, ...]
```

✅ *This creates a vector representation of the text to enable similarity search.*  

---

### 📌 **Vectors**  
- Numerical arrays representing embeddings.  
- Used to measure similarity between data points.  

**Example:**  
👉 `"Paris is the capital of France"` → `[0.12, -0.45, 0.98, -0.21, ...]`  
👉 `"Berlin is the capital of Germany"` → `[0.15, -0.43, 0.91, -0.20, ...]`  

✅ *Vectors represent the semantic meaning of text, allowing for comparison.*  

---

### 📌 **Similarity Search**  
- Process of finding the most relevant embeddings based on a similarity score.  
- Techniques:  
  - **Cosine Similarity**  
  - **Euclidean Distance**  

**Example:**  
```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

vec1 = np.array([0.12, -0.45, 0.98])
vec2 = np.array([0.15, -0.43, 0.91])

similarity = cosine_similarity([vec1], [vec2])
print(similarity)  # Output: [[0.998]]
```

✅ *Higher similarity score = more similar content.*  

---

### 📌 **Cosine Similarity**  
Measures the angle between two vectors.  
\[
\text{similarity}(A, B) = \frac{A \cdot B}{||A|| \times ||B||}
\]

✅ *Cosine similarity is commonly used because it measures semantic similarity accurately.*  

---

### 📌 **Chunking**  
- Splitting large text into smaller parts for better embedding and retrieval.  

**Example:**  
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text = "Neo4j is a graph database that stores data as nodes and edges..."
splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
chunks = splitter.split_text(text)
print(chunks)
```

✅ *Chunking allows handling large documents efficiently.*  

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

✅ *Tokenization helps LLM understand and process text.*  

---

## 💾 **SQL vs NoSQL vs Graph DB**  

| Feature | SQL (Relational) | NoSQL (Document-based) | Graph DB (Neo4j) |
|---------|-------------------|------------------------|------------------|
| **Data Structure** | Tables (Rows & Columns) | Collections & Documents | Nodes & Relationships |
| **Use Case** | Financial data, structured data | Unstructured data | Connected data |
| **Performance** | Slower for complex joins | Fast for large data | Fast for connected data |
| **Schema** | Rigid | Flexible | Flexible |
| **Example** | MySQL, PostgreSQL | MongoDB, Firebase | Neo4j, ArangoDB |

✅ *Graph DB excels in handling relationships between data points.*  

---

## 📚 **How Graph RAG Works with Neo4j**  

### ✅ **Graph Structure in Neo4j**  
- **Nodes** → Entities (documents, authors, keywords)  
- **Relationships** → Contextual connections between nodes  

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

✅ *This sets up the connection to a Neo4j instance.*  

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

✅ *This creates a new document node with embeddings.*  

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

✅ *This retrieves the most relevant documents.*  

---

### **Step 4: Generate Response with LLM**  
```python
from langchain.llms import OpenAI

llm = OpenAI()
prompt = f"Using the following context:\n{result}\nExplain Neo4j."
response = llm(prompt)
print(response)
```

✅ *This generates an answer based on retrieved context.*  

---

## 🌟 **Why Graph RAG is Better**  
- Faster retrieval  
- Better contextual understanding  
- Stronger relationship-based search  
- Handles real-time data better  

---

# 🌐 **LangGraph – Hands-On Training**  

---

## 🏆 **Objectives**  
✅ Understand LangGraph and its key concepts  
✅ Explain how LangGraph enables multi-step workflows  
✅ Create an example LangGraph-based RAG pipeline  
✅ Handle memory, state, and complex branching in workflows  
✅ Integrate LangGraph with Neo4j for Graph RAG  

---

## 🌍 **1. What is LangGraph?**  
LangGraph is a **graph-based framework** built on top of **LangChain** that enables the creation of **multi-step, multi-agent workflows** for LLM-based applications.  

---

### 🚀 **LangGraph = LangChain + State Management + Graph-Based Flow**  
1. **LangChain** → Framework for building LLM-based applications  
2. **State Management** → Maintains memory and context across multiple steps  
3. **Graph-Based Flow** → Directed graph (DAG) model to control execution path  

---

### 💡 **Why LangGraph Over LangChain?**  
| Feature | LangChain | LangGraph |
|---|---|---|
| **Single-step execution** | ✅ | ✅ |
| **Multi-step execution** | ❌ | ✅ |
| **Parallelism** | ❌ | ✅ |
| **Branching & Conditional Logic** | ❌ | ✅ |
| **Stateful Memory** | Limited | ✅ |
| **Event Handling** | ❌ | ✅ |

---

## 🔍 **2. Key Concepts in LangGraph**  

---

### 📌 **(i) Nodes**  
- Building blocks of a LangGraph workflow  
- Represents **steps** in the pipeline  
- Can be:  
  - LLM calls  
  - Embedding generation  
  - Data retrieval  
  - Similarity search  

✅ **Example:**  
- Node 1 → Generate embedding  
- Node 2 → Search Neo4j  
- Node 3 → Generate output  

---

### 📌 **(ii) Edges**  
- Connect nodes together  
- Represent flow of data between steps  
- Can be **conditional** or **unconditional**  

✅ **Example:**  
- If similarity > 0.8 → Use result for augmentation  
- If similarity < 0.8 → Search another source  

---

### 📌 **(iii) State**  
- Shared memory across nodes  
- Tracks intermediate results  
- Supports complex reasoning  

✅ **Example:**  
- Store query embeddings  
- Track search results  
- Pass retrieved documents to LLM  

---

### 📌 **(iv) Branching**  
- Create different paths based on conditions  
- Handles **multi-hop reasoning**  
- Supports real-time adjustments  

✅ **Example:**  
- If data is not found → Perform web search  
- If data is found → Augment response  

---

### 📌 **(v) Parallel Execution**  
- Execute multiple nodes simultaneously  
- Improves efficiency and reduces latency  

✅ **Example:**  
- Generate embeddings while searching Neo4j  
- Search multiple sources in parallel  

---

### 📌 **(vi) Memory**  
- Stateful memory persists across nodes  
- Provides continuity and context retention  
- Avoids repetitive queries  

✅ **Example:**  
- Track previous conversation history  
- Remember user preferences  

---

## 🏗️ **3. Example: LangGraph-Based RAG with Neo4j**  

---

### ✅ **Step 1: Install Dependencies**  
```bash
pip install langchain langgraph neo4j openai
```

---

### ✅ **Step 2: Import Dependencies**  
```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain.schema import Document
from neo4j import GraphDatabase
```

---

### ✅ **Step 3: Define State**  
- Define memory for sharing data between nodes  

```python
class RAGState:
    query: str
    retrieved_docs: list
    final_answer: str
```

---

### ✅ **Step 4: Set Up Neo4j Connection**  
```python
handler = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
```

---

### ✅ **Step 5: Define Node 1 – Generate Embedding**  
```python
embeddings = OpenAIEmbeddings()

def generate_embedding(state):
    query = state.query
    vector = embeddings.embed_query(query)
    return {"query_vector": vector}
```

---

### ✅ **Step 6: Define Node 2 – Search Neo4j**  
```python
def search_neo4j(state):
    query_vector = state["query_vector"]
    
    cypher_query = """
    MATCH (d:Document)
    RETURN d.text, 
           cosineSimilarity(d.embedding, $query_vector) AS score
    ORDER BY score DESC
    LIMIT 3
    """
    
    with handler.session() as session:
        results = session.run(cypher_query, query_vector=query_vector)
        docs = [record["d.text"] for record in results]
    
    return {"retrieved_docs": docs}
```

---

### ✅ **Step 7: Define Node 3 – Generate Answer**  
```python
llm = ChatOpenAI()

def generate_answer(state):
    context = "\n".join(state["retrieved_docs"])
    prompt = f"Context:\n{context}\n\nQuestion: {state['query']}\nAnswer:"
    
    response = llm.predict(prompt)
    
    return {"final_answer": response}
```

---

### ✅ **Step 8: Build Graph**  
```python
graph = StateGraph(RAGState)

# Add nodes
graph.add_node("generate_embedding", generate_embedding)
graph.add_node("search_neo4j", search_neo4j)
graph.add_node("generate_answer", generate_answer)

# Define edges
graph.add_edge("generate_embedding", "search_neo4j")
graph.add_edge("search_neo4j", "generate_answer")
graph.add_edge("generate_answer", END)

# Set entry point
graph.set_entry_point("generate_embedding")

# Compile graph
app = graph.compile()
```

---

### ✅ **Step 9: Execute Workflow**  
```python
query = "What is Graph RAG?"
state = RAGState(query=query)

# Run pipeline
result = app.invoke(state)
print(result["final_answer"])
```

---

## 🧠 **4. How LangGraph Handles Complex Logic**  
✅ **Multi-hop Reasoning:**  
- Search graph → Augment with context → Generate response  

✅ **Real-time Adjustments:**  
- Handle missing context dynamically  

✅ **Efficient Retrieval:**  
- Search multiple sources simultaneously  

✅ **State Preservation:**  
- LLM maintains context across nodes  

---

## 🌟 **5. Why LangGraph + Graph RAG = 💪**  
| Challenge | Solution |
|---|---|
| Context length limits | External knowledge retrieval using graph search |
| Hallucination | Fact-based generation using grounded data |
| Complex queries | Multi-hop search across graph nodes |
| Real-time updates | Graph-based search handles dynamic changes |
| Data relationships | Graph model preserves entity relationships |

---

## 🎯 **6. Multi-Modal Expansion**  
✅ Text + Images + Videos + Audio  
✅ Handle diverse data types using vector embeddings  

### ✅ **Example:**  
1. Text → OpenAI embeddings  
2. Image → CLIP embeddings  
3. Audio → Whisper embeddings  

---

### ✅ **Example Code:**  
```python
from langchain.embeddings import CLIPEmbeddings

clip = CLIPEmbeddings()
image_vector = clip.embed_image("path/to/image.jpg")
```

---

## 🚀 **7. Why LangGraph Over Traditional Pipelines**  
✅ Easy to build complex workflows  
✅ Better branching and error handling  
✅ Parallel execution reduces latency  
✅ State management improves consistency  

---

## 🏆 **8. Summary**  
🚀 LangGraph + Graph RAG = **Best of Both Worlds**  
💡 Multi-step, Multi-hop, Parallel Reasoning  
🔎 Neo4j provides **structured graph search**  
🤖 OpenAI provides **high-quality text generation**  

---

## 🙌 **Q&A**  

