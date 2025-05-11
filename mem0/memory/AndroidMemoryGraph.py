# -*- coding: utf-8 -*-
"""
Created on Sun May 11 11:55:15 2025

@author: edoardo
"""
import numpy as np
import hnswlib
from typing import List, Dict
from sentence_transformers import SentenceTransformer

from mem0.graphs.tools import (
    DELETE_MEMORY_STRUCT_TOOL_GRAPH,
    DELETE_MEMORY_TOOL_GRAPH,
    EXTRACT_ENTITIES_STRUCT_TOOL,
    EXTRACT_ENTITIES_TOOL,
    RELATIONS_STRUCT_TOOL,
    RELATIONS_TOOL,
)
from mem0.graphs.utils import EXTRACT_RELATIONS_PROMPT, get_delete_messages
from mem0.utils.factory import EmbedderFactory, LlmFactory

logger = logging.getLogger(__name__)

class AndroidMemoryGraph:
    def __init__(self, config):
        self.config = config
        self.db_path = config.graph_store.config.get("path", ":memory:")
        self.conn = sqlite3.connect(self.db_path)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim embeddings
        self.embedding_dim = 384
        self.threshold = 0.7
        
        # Initialize vector index
        self.index = hnswlib.Index(space="cosine", dim=self.embedding_dim)
        self.index.init_index(max_elements=1000)
        self.node_ids = []  # Maps index positions to node IDs
        
        # Initialize database
        self._init_db()
        self._init_vector_index()
        """
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            {"enable_embeddings": True},
        )

        self.llm_provider = "llama"
        if self.config.llm.provider:
            self.llm_provider = self.config.llm.provider
        if self.config.graph_store.llm:
            self.llm_provider = self.config.graph_store.llm.provider

        self.llm = LlmFactory.create(self.llm_provider, self.config.llm.config)
        self.user_id = None
        self.threshold = 0.7
      """

    # === Core Database Methods ===
    def _init_db(self):
        """Initialize SQLite tables with proper indexes"""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY,
                name TEXT,
                type TEXT,
                embedding BLOB,
                user_id TEXT
            );
            CREATE TABLE IF NOT EXISTS relationships (
                source_id INTEGER,
                target_id INTEGER,
                type TEXT,
                user_id TEXT,
                PRIMARY KEY (source_id, target_id, type),
                FOREIGN KEY (source_id) REFERENCES nodes(id),
                FOREIGN KEY (target_id) REFERENCES nodes(id)
            );
            CREATE INDEX IF NOT EXISTS idx_nodes_user ON nodes(user_id);
            CREATE INDEX IF NOT EXISTS idx_rels_user ON relationships(user_id);
        """)
        self.conn.commit()

    def _init_vector_index(self):
        """Load existing embeddings into HNSW (hierarchical navigable small world) index
           # execute SQL query on databse connection
           # retrieves all records with ID and embedding columns from the nodes tables
           # return cur object to interate thourgh the results
           # iterate through each rows and unpack the values into node_id and emb_blob
           # convert binary blob back to array of floats using numpy
           # maintains a mapping between index positions and original node IDs
        
        
        """
        cur = self.conn.execute("SELECT id, embedding FROM nodes")
        for node_id, emb_blob in cur:
            embedding = np.frombuffer(emb_blob, dtype=np.float32)
            self.index.add_items(np.array([embedding]), np.array([node_id]))
            self.node_ids.append(node_id)

    # === CRUD Operations ===
    def add(self, data: str, filters: dict) -> dict:
        """
        Add data to the knowledge graph
        Args:
            data: Text containing entities/relationships
            filters: {"user_id": "user123"}
        Returns:
            {"added_entities": List[dict], "deleted_entities": List[dict]}
        """
        entity_type_map = self._retrieve_nodes_from_data(data, filters)
        to_be_added = self._establish_nodes_relations_from_data(data, filters, entity_type_map)
        
        # Add nodes and relationships
        added_entities = []
        for item in to_be_added:
            source_id = self._get_or_create_node(
                item["source"], 
                entity_type_map.get(item["source"], "Entity"), 
                filters["user_id"]
            )
            target_id = self._get_or_create_node(
                item["destination"], 
                entity_type_map.get(item["destination"], "Entity"), 
                filters["user_id"]
            )
            
            self.conn.execute(
                "INSERT OR IGNORE INTO relationships VALUES (?, ?, ?, ?)",
                (source_id, target_id, item["relationship"], filters["user_id"])
            )
            
            added_entities.append({
                "source": item["source"],
                "relationship": item["relationship"],
                "target": item["destination"]
            })
        
        self.conn.commit()
        return {"added_entities": added_entities, "deleted_entities": []}

    def search(self, query: str, filters: dict, limit: int = 5) -> List[dict]:
        """Semantic search with vector similarity"""
        query_embedding = self.embedding_model.encode(query)
        ids, distances = self.index.knn_query(query_embedding, k=limit)
        
        results = []
        for node_id, distance in zip(ids[0], distances[0]):
            # Get relationships for matched nodes
            cur = self.conn.execute("""
                SELECT n1.name, r.type, n2.name
                FROM relationships r
                JOIN nodes n1 ON r.source_id = n1.id
                JOIN nodes n2 ON r.target_id = n2.id
                WHERE (r.source_id = ? OR r.target_id = ?) 
                AND r.user_id = ?
                """, (node_id, node_id, filters["user_id"]))
            
            for source, rel_type, target in cur:
                results.append({
                    "source": source,
                    "relationship": rel_type,
                    "target": target,
                    "score": float(1 - distance)
                })
        
        return results[:limit]

    def delete_all(self, filters: dict):
        """Delete all data for a user"""
        # Clear from vector index
        cur = self.conn.execute("SELECT id FROM nodes WHERE user_id = ?", (filters["user_id"],))
        for (node_id,) in cur:
            try:
                idx = self.node_ids.index(node_id)
                self.index.mark_deleted(idx)
            except ValueError:
                pass
        
        # Clear database
        self.conn.execute("DELETE FROM relationships WHERE user_id = ?", (filters["user_id"],))
        self.conn.execute("DELETE FROM nodes WHERE user_id = ?", (filters["user_id"],))
        self.conn.commit()

    def get_all(self, filters: dict, limit: int = 100) -> List[dict]:
        """Get all relationships for a user"""
        cur = self.conn.execute("""
            SELECT n1.name, r.type, n2.name
            FROM relationships r
            JOIN nodes n1 ON r.source_id = n1.id
            JOIN nodes n2 ON r.target_id = n2.id
            WHERE r.user_id = ?
            LIMIT ?
            """, (filters["user_id"], limit))
        
        return [{
            "source": row[0],
            "relationship": row[1],
            "target": row[2]
        } for row in cur]

    # === Helper Methods ===
    def _get_or_create_node(self, name: str, node_type: str, user_id: str) -> int:
        """Get existing node or create new with embedding"""
        cur = self.conn.execute(
            "SELECT id FROM nodes WHERE name = ? AND user_id = ?", 
            (name, user_id))
        if row := cur.fetchone():
            return row[0]
        
        # Create new node
        embedding = self.embedding_model.encode(name)
        cur = self.conn.execute(
            "INSERT INTO nodes (name, type, embedding, user_id) VALUES (?, ?, ?, ?)",
            (name, node_type, embedding.tobytes(), user_id))
        node_id = cur.lastrowid
        
        # Update index
        self.index.add_items(np.array([embedding]), np.array([node_id]))
        self.node_ids.append(node_id)
        return node_id

    


    def _retrieve_nodes_from_data(self, data, filters):
      """Extracts entities using LLaMA (local LLM on Android)."""
      # System prompt for entity extraction that can be added to the database
      system_prompt = f"""
      You are an entity extraction assistant. Identify entities and their types from the text.
      - Replace 'I', 'me', 'my' with user_id: {filters['user_id']}.
      - Return entities in JSON format: {{"entity": "<name>", "type": "<type>"}}.
      - Types must be lowercase (e.g., 'person', 'organization').
      - Ignore questions; only extract entities.
      """

      # Generate response using LLaMA
      response = self.llm.generate_response(
          messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": data}
          ],
          max_tokens=512,  # Limit for mobile
      )

      # Parse LLaMA's response (assuming JSON output)
      entity_type_map = {}
      try:
          # Extract JSON from LLaMA's response (adjust based on your LLM's output format)
          entities_json = self._extract_json_from_llm_response(response)
          for item in entities_json:
              entity = item["entity"].lower().replace(" ", "_")
              entity_type = item["type"].lower().replace(" ", "_")
              entity_type_map[entity] = entity_type
      except Exception as e:
          logger.error(f"LLaMA entity extraction failed: {e}")
          return {}  # Fallback: Return empty dict or use regex/NER as backup

      logger.debug(f"Entity type map: {entity_type_map}")
      return entity_type_map

    def _parse_llm_relationship_response(self, response):
      """Extract JSON from LLaMA's text response (e.g., '```json\n...```')."""
      import re
      import json

      # Extract JSON block (common in LLM responses)
      json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
      if json_match:
          return json.loads(json_match.group(1))
      
      # Fallback: Assume raw JSON string
      return json.loads(response)


    def _establish_nodes_relations_from_data(self, data, filters, entity_type_map):
      """Extract relationships using a local LLM (LLaMA) on Android."""
      # 1. Prepare the system prompt (structured for JSON output)
      system_prompt = f"""
      You are a relationship extraction assistant. Identify relationships between entities in the text.
      - Replace 'I', 'me', 'my' with user_id: {filters['user_id']}.
      - Use the entity list: {list(entity_type_map.keys())}.
      - Return relationships in JSON format: 
        [{{"source": "entity1", "relationship": "relation", "destination": "entity2"}}].
      - Example:
        Input: "Apple makes the iPhone."
        Output: [{{"source": "apple", "relationship": "makes", "destination": "iphone"}}].
      """
      
      # 2. Generate response using LLaMA
      response = self.llm.generate_response(
          messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": data}
          ],
          max_tokens=512,  # Limit for mobile
      )

      # 3. Parse LLaMA's JSON output (fallback to regex if needed)
      try:
          entities = self._parse_llm_relationship_response(response)
          entities = self._remove_spaces_from_entities(entities)  # Normalize
      except Exception as e:
          logger.error(f"LLaMA relationship extraction failed: {e}")
          entities = []  # Fallback to empty list

      logger.debug(f"Extracted relationships: {entities}")
      return entities









    def _search_graph_db(self, node_list, filters, limit=100):
      """Search similar nodes and their relationships using SQLite + HNSW."""
      result_relations = []
      
      for node in node_list:
          # 1. Generate embedding for the query node
          n_embedding = self.embedding_model.encode(node)
          
          # 2. Find similar nodes via HNSW index
          similar_ids, distances = self.index.knn_query(np.array([n_embedding]), k=limit)
          similar_ids = similar_ids[0]  # Remove batch dimension
          distances = distances[0]
          
          # 3. Process each similar node
          for node_id, distance in zip(similar_ids, distances):
              similarity = 1 - distance  # Convert distance to similarity
              
              if similarity < self.threshold:
                  continue  # Skip below-threshold matches
                  
              # 4. Get outgoing relationships (source -> target)
              outgoing = self.conn.execute("""
                  SELECT n1.name, r.type, n2.name
                  FROM relationships r
                  JOIN nodes n1 ON r.source_id = n1.id
                  JOIN nodes n2 ON r.target_id = n2.id
                  WHERE r.source_id = ? AND r.user_id = ?
                  LIMIT ?
              """, (node_id, filters["user_id"], limit)).fetchall()
              
              # 5. Get incoming relationships (target <- source)
              incoming = self.conn.execute("""
                  SELECT n1.name, r.type, n2.name
                  FROM relationships r
                  JOIN nodes n1 ON r.source_id = n1.id
                  JOIN nodes n2 ON r.target_id = n2.id
                  WHERE r.target_id = ? AND r.user_id = ?
                  LIMIT ?
              """, (node_id, filters["user_id"], limit)).fetchall()
              
              # 6. Format results with similarity scores
              for src_name, rel_type, tgt_name in outgoing:
                  result_relations.append({
                      "source": src_name,
                      "relationship": rel_type,
                      "destination": tgt_name,
                      "similarity": float(similarity)
                  })
                  
              for src_name, rel_type, tgt_name in incoming:
                  result_relations.append({
                      "source": src_name,
                      "relationship": rel_type,
                      "destination": tgt_name,
                      "similarity": float(similarity)
                  })
      
      # 7. Sort by similarity and remove duplicates
      unique_relations = {}
      for rel in result_relations:
          key = (rel["source"], rel["relationship"], rel["destination"])
          if key not in unique_relations or rel["similarity"] > unique_relations[key]["similarity"]:
              unique_relations[key] = rel
      
      return sorted(unique_relations.values(), key=lambda x: -x["similarity"])[:limit]   


    def _get_delete_entities_from_search_output(self, search_output, data, filters):
      """Get entities to delete using local LLM (LLaMA) without OpenAI tool calls."""
      ##TODO


      

    def _remove_spaces_from_entities(self, entities: List[dict]) -> List[dict]:
        """Normalize entity names"""
        for item in entities:
            item["source"] = item["source"].lower().replace(" ", "_")
            item["relationship"] = item["relationship"].lower().replace(" ", "_")
            item["destination"] = item["destination"].lower().replace(" ", "_")
        return entities

    # === Advanced Methods ===
    def batch_add(self, data_list: List[str], filters: dict):
        """Optimized batch insert"""
        with self.conn:
            for data in data_list:
                self.add(data, filters)

    def export_to_file(self, filepath: str):
        """Export database to file"""
        if self.db_path == ":memory:":
            with sqlite3.connect(filepath) as dest:
                self.conn.backup(dest)

    def close(self):
        """Cleanup resources"""
        self.conn.close()
        self.index = None