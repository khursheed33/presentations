# Retrieval-Augmented Generation (RAG) Overview

RAG is a technique that combines retrieval systems with generative AI models to enhance the quality and factuality of generated responses. Let me explain RAG and the key concepts you mentioned:

## Core RAG Concepts

### Embeddings & Vectors
Embeddings are dense vector representations of text that capture semantic meaning. When text is converted to embeddings, similar concepts are positioned close together in vector space.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
text = "Artificial intelligence is transforming industries."
embedding = model.encode(text)  # Creates a vector representation
```

### Similarity Search
Finding relevant documents by comparing vector representations using distance metrics.

### Cosine Similarity
A measure of similarity between two vectors that calculates the cosine of the angle between them. Values range from -1 (opposite) to 1 (identical).

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

similarity = cosine_similarity([embedding1], [embedding2])[0][0]
```

### Chunking
Breaking down documents into smaller, manageable pieces for processing and retrieval.

```python
def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
```

### Tokens
Basic units of text (words or subwords) that language models process.

## RAG with Neo4j: Practical Example

```python
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import openai

class Neo4jRAG:
    def __init__(self, uri, username, password, embedding_model="all-MiniLM-L6-v2"):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.embedding_model = SentenceTransformer(embedding_model)
        self.openai_client = openai.OpenAI()
    
    def close(self):
        self.driver.close()
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a piece of text"""
        return self.embedding_model.encode(text).tolist()
    
    def chunk_document(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split a document into chunks with overlap"""
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk:
                chunks.append(chunk)
        return chunks
    
    def store_document(self, doc_id: str, title: str, content: str, metadata: Dict = None):
        """Store a document in Neo4j with embeddings"""
        chunks = self.chunk_document(content)
        
        with self.driver.session() as session:
            # Create document node
            session.run(
                """
                CREATE (d:Document {id: $id, title: $title, metadata: $metadata})
                """,
                id=doc_id, title=title, metadata=metadata or {}
            )
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                embedding = self.generate_embedding(chunk)
                
                # Store chunk with embedding
                session.run(
                    """
                    MATCH (d:Document {id: $doc_id})
                    CREATE (c:Chunk {
                        id: $chunk_id,
                        content: $content,
                        embedding: $embedding,
                        chunk_number: $chunk_number
                    })
                    CREATE (d)-[:HAS_CHUNK]->(c)
                    """,
                    doc_id=doc_id, 
                    chunk_id=f"{doc_id}_chunk_{i}",
                    content=chunk,
                    embedding=embedding,
                    chunk_number=i
                )
    
    def create_vector_index(self):
        """Create a vector index for similarity search"""
        with self.driver.session() as session:
            # Drop existing index if it exists
            session.run("DROP INDEX vector_index IF EXISTS")
            
            # Create new vector index
            session.run(
                """
                CREATE VECTOR INDEX vector_index FOR (c:Chunk) 
                ON (c.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 384,
                    `vector.similarity_function`: 'cosine'
                }}
                """
            )
    
    def retrieve_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve most similar chunks to the query"""
        query_embedding = self.generate_embedding(query)
        
        with self.driver.session() as session:
            result = session.run(
                """
                CALL db.index.vector.queryNodes('vector_index', $top_k, $query_embedding)
                YIELD node, score
                WITH node, score
                MATCH (d:Document)-[:HAS_CHUNK]->(node)
                RETURN d.title AS document_title,
                       node.content AS chunk_content,
                       score AS similarity_score
                ORDER BY similarity_score DESC
                """,
                top_k=top_k,
                query_embedding=query_embedding
            )
            
            return [dict(record) for record in result]
    
    def query_with_rag(self, user_query: str, top_k: int = 3) -> str:
        """Perform RAG: retrieve relevant chunks and generate response"""
        # 1. Retrieve similar chunks
        retrieved_chunks = self.retrieve_similar_chunks(user_query, top_k)
        
        # 2. Prepare context for the LLM
        context = "\n\n".join([f"Document: {chunk['document_title']}\nContent: {chunk['chunk_content']}" 
                             for chunk in retrieved_chunks])
        
        # 3. Prepare the prompt
        prompt = f"""
        Answer the following question based on the provided context.
        If you cannot answer the question based on the context, say "I don't have enough information to answer this question."
        
        Context:
        {context}
        
        Question: {user_query}
        """
        
        # 4. Generate response using OpenAI
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def create_knowledge_graph(self):
        """Extract entities and relationships from chunks to build a knowledge graph"""
        with self.driver.session() as session:
            chunks = session.run("MATCH (c:Chunk) RETURN c.id AS id, c.content AS content")
            
            for chunk in chunks:
                # Use LLM to extract entities and relationships
                prompt = f"""
                Extract the main entities and relationships from the following text.
                Format your response as a JSON object with two arrays: 'entities' and 'relationships'.
                Each entity should have a 'name' and 'type'.
                Each relationship should have a 'source', 'target', and 'type'.
                
                Text: {chunk['content']}
                """
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a system that extracts structured knowledge from text."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2
                )
                
                try:
                    import json
                    extraction = json.loads(response.choices[0].message.content)
                    
                    # Create entities
                    for entity in extraction.get('entities', []):
                        session.run(
                            """
                            MERGE (e:Entity {name: $name, type: $type})
                            """,
                            name=entity['name'], type=entity['type']
                        )
                    
                    # Create relationships
                    for rel in extraction.get('relationships', []):
                        session.run(
                            """
                            MATCH (source:Entity {name: $source})
                            MATCH (target:Entity {name: $target})
                            MERGE (source)-[:RELATES_TO {type: $type}]->(target)
                            """,
                            source=rel['source'], target=rel['target'], type=rel['type']
                        )
                    
                    # Connect chunk to entities
                    for entity in extraction.get('entities', []):
                        session.run(
                            """
                            MATCH (c:Chunk {id: $chunk_id})
                            MATCH (e:Entity {name: $entity_name, type: $entity_type})
                            MERGE (c)-[:MENTIONS]->(e)
                            """,
                            chunk_id=chunk['id'], entity_name=entity['name'], entity_type=entity['type']
                        )
                except Exception as e:
                    print(f"Error processing chunk {chunk['id']}: {str(e)}")
    
    def query_knowledge_graph(self, query: str) -> str:
        """Query the knowledge graph for information"""
        # Generate embedding for the query
        query_embedding = self.generate_embedding(query)
        
        with self.driver.session() as session:
            # Find relevant chunks
            result = session.run(
                """
                CALL db.index.vector.queryNodes('vector_index', 3, $query_embedding)
                YIELD node AS chunk, score
                MATCH (chunk)-[:MENTIONS]->(entity)
                RETURN entity.name AS entity_name, entity.type AS entity_type, 
                       collect(chunk.content) AS relevant_chunks, score
                ORDER BY score DESC
                LIMIT 5
                """,
                query_embedding=query_embedding
            )
            
            entities = [dict(record) for record in result]
            
            if not entities:
                return "No relevant information found in the knowledge graph."
            
            # Get relationships between top entities
            top_entities = [entity['entity_name'] for entity in entities[:3]]
            relationships = session.run(
                """
                MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity)
                WHERE e1.name IN $entities AND e2.name IN $entities
                RETURN e1.name AS source, e2.name AS target, r.type AS relationship_type
                """,
                entities=top_entities
            )
            
            # Format response
            response = "Knowledge Graph Query Results:\n\n"
            response += "Top Entities:\n"
            for entity in entities:
                response += f"- {entity['entity_name']} ({entity['entity_type']})\n"
            
            response += "\nRelationships:\n"
            for rel in relationships:
                response += f"- {rel['source']} -> {rel['relationship_type']} -> {rel['target']}\n"
            
            return response

# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = Neo4jRAG(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password"
    )
    
    # Store a document
    rag.store_document(
        doc_id="doc1",
        title="Introduction to Graph Databases",
        content="Graph databases store data in nodes and relationships, making them ideal for connected data. Neo4j is a popular graph database that uses the Cypher query language. Graph databases excel at tasks like recommendation engines, fraud detection, and knowledge graphs."
    )
    
    # Create vector index
    rag.create_vector_index()
    
    # Query with RAG
    result = rag.query_with_rag("What are the applications of graph databases?")
    print(result)
    
    # Create knowledge graph
    rag.create_knowledge_graph()
    
    # Query knowledge graph
    kg_result = rag.query_knowledge_graph("Neo4j applications")
    print(kg_result)
    
    # Close connection
    rag.close()

```

## Graph RAG

Graph RAG extends traditional RAG by incorporating graph structures for more contextual retrieval. Instead of just retrieving similar chunks, Graph RAG uses relationships between entities to provide more comprehensive and contextually relevant information.

### Advantages of Graph RAG:
1. Captures relationships between information pieces
2. Provides context-aware recommendations
3. Offers multi-hop reasoning capabilities
4. Handles complex queries requiring connected information

### LangGraph

LangGraph is a framework for building LLM-powered applications with a graph-based architecture. It allows you to create stateful, reasoning-enabled applications by organizing LLM calls and other functions into a directed graph.

## Effective RAG Prompts

Here are 10 well-structured prompts for working with RAG and Neo4j:



# 10 Effective RAG Prompts for Neo4j Graph Database

1. **Knowledge Graph Creation**: "Extract entities and relationships from the following text about {topic}, and create a knowledge graph in Neo4j representing these connections."

2. **Multi-hop Reasoning**: "Using the knowledge graph, find all connections between {entity1} and {entity2} within 3 hops, and explain the significance of each path."

3. **Contextual Recommendation**: "Based on the user's interaction with {entity}, recommend related entities from the knowledge graph and explain why they're relevant."

4. **Anomaly Detection**: "Analyze this transaction data and identify unusual patterns using graph algorithms. Return Neo4j Cypher queries to visualize potential anomalies."

5. **Domain-Specific QA**: "Answer the following question about {domain} using information from the knowledge graph: {question}. Include relevant entities and relationships in your response."

6. **Information Retrieval Enhancement**: "Enhance this search query by expanding it to include related concepts from the knowledge graph: {query}"

7. **Graph-Based Summarization**: "Summarize the key information about {topic} by identifying the most central entities and their relationships in the knowledge graph."

8. **Knowledge Base Completion**: "Identify missing relationships in the knowledge graph related to {entity} and suggest Cypher queries to add them."

9. **Trend Analysis**: "Analyze temporal patterns in the knowledge graph related to {topic} over the past {timeframe} and generate insights."

10. **Causal Reasoning**: "Using the knowledge graph, identify potential causal relationships between {event1} and {event2}, and explain your reasoning."


## Implementation Considerations

When implementing RAG with Neo4j:

1. **Indexing**: Create proper vector indexes for efficient similarity search
2. **Schema Design**: Plan your graph schema carefully to represent relationships effectively
3. **Query Efficiency**: Optimize Cypher queries for performance
4. **Embedding Selection**: Choose appropriate embedding models for your domain
5. **Chunking Strategy**: Determine optimal chunk sizes and overlap for your content
