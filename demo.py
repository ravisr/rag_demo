# Complete Code Completion Implementation for TOSCA-based Infrastructure Automation Tool
# Using RAG+Graph RAG & MCP Server

import os
import json
import yaml
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from fastapi import FastAPI, Request, HTTPException
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.llms import OpenAI  # You can use any LLM here
from langchain.prompts import PromptTemplate
import uvicorn

# Configuration and constants
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
EMBED_MODEL = "text-embedding-ada-002"  # Or your preferred embedding model
LLM_MODEL = "gpt-4" # Or your preferred LLM model
YAML_EXTENSIONS = [".yml", ".yaml"]
VECTOR_DB_PATH = "vector_store"
GRAPH_DB_PATH = "graph_store.gpickle"

# Initialize API
app = FastAPI(title="TOSCA Code Completion API")

# Data structures
@dataclass
class FileContext:
    """Represents context information from a file."""
    file_path: str
    content: str
    line_number: int
    column_number: int
    prefix: str  # Text before cursor
    suffix: str  # Text after cursor

@dataclass
class CompletionCandidate:
    """Represents a possible code completion."""
    text: str
    score: float
    source: str  # e.g., "semantic", "graph", "structural"

@dataclass
class CompletionResponse:
    """API response for completion requests."""
    candidates: List[CompletionCandidate]
    context_used: Dict[str, Any]

# 1. Traditional RAG Implementation
class SemanticRAG:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
        self.vector_store = None

    def index_documents(self, folder_path: str) -> None:
        """Index YAML files from the provided folder."""
        # Load all YAML files from the folder
        yaml_loader = DirectoryLoader(
            folder_path, 
            glob="**/*.y*ml",
            loader_cls=TextLoader
        )
        documents = yaml_loader.load()
        
        # Split documents into chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n---\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create or update vector store
        if os.path.exists(VECTOR_DB_PATH):
            self.vector_store = FAISS.load_local(VECTOR_DB_PATH, self.embeddings)
            self.vector_store.add_documents(chunks)
        else:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            self.vector_store.save_local(VECTOR_DB_PATH)
        
        print(f"Indexed {len(chunks)} chunks from {len(documents)} documents")

    def retrieve_similar(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve semantically similar chunks to the query."""
        if not self.vector_store:
            if os.path.exists(VECTOR_DB_PATH):
                self.vector_store = FAISS.load_local(VECTOR_DB_PATH, self.embeddings)
            else:
                raise ValueError("Vector store not initialized. Run index_documents first.")
        
        return self.vector_store.similarity_search(query, k=k)

# 2. Graph RAG Implementation
class GraphRAG:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.yaml_structures = {}
    
    def parse_yaml_to_graph(self, folder_path: str) -> None:
        """Parse YAML files and build a graph representation."""
        yaml_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if any(file.endswith(ext) for ext in YAML_EXTENSIONS):
                    yaml_files.append(os.path.join(root, file))
        
        # Build graph from YAML files
        for file_path in yaml_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    yaml_dict = yaml.safe_load(content)
                    self.yaml_structures[file_path] = yaml_dict
                    self._build_graph_from_yaml(yaml_dict, file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Save graph
        nx.write_gpickle(self.graph, GRAPH_DB_PATH)
        print(f"Built graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")

    def _build_graph_from_yaml(self, yaml_dict: Dict, file_path: str, parent_path: str = "") -> None:
        """Recursively build graph from YAML dictionary."""
        if not isinstance(yaml_dict, dict):
            return
        
        for key, value in yaml_dict.items():
            current_path = f"{parent_path}.{key}" if parent_path else key
            node_id = f"{file_path}:{current_path}"
            
            # Add node with metadata
            self.graph.add_node(
                node_id,
                key=key,
                file_path=file_path,
                path=current_path,
                value_type=type(value).__name__
            )
            
            # Add edge from parent to current node
            if parent_path:
                parent_node_id = f"{file_path}:{parent_path}"
                self.graph.add_edge(parent_node_id, node_id, relation="contains")
            
            # Process nested structures
            if isinstance(value, dict):
                self._build_graph_from_yaml(value, file_path, current_path)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        item_path = f"{current_path}[{i}]"
                        item_node_id = f"{file_path}:{item_path}"
                        self.graph.add_node(
                            item_node_id,
                            key=f"{key}[{i}]",
                            file_path=file_path,
                            path=item_path,
                            value_type=type(item).__name__
                        )
                        self.graph.add_edge(node_id, item_node_id, relation="contains")
                        self._build_graph_from_yaml(item, file_path, item_path)
            
            # Add connections between similar nodes across files
            if isinstance(value, (str, int, float, bool)):
                # Find similar nodes with same key
                similar_nodes = [n for n in self.graph.nodes if self.graph.nodes[n].get('key') == key]
                for similar_node in similar_nodes:
                    if similar_node != node_id:
                        self.graph.add_edge(node_id, similar_node, relation="similar_key")

    def get_structural_context(self, file_path: str, yaml_path: str) -> List[Tuple[str, Any]]:
        """Get structural context based on graph relations."""
        if not self.graph:
            if os.path.exists(GRAPH_DB_PATH):
                self.graph = nx.read_gpickle(GRAPH_DB_PATH)
            else:
                raise ValueError("Graph not initialized. Run parse_yaml_to_graph first.")
        
        node_id = f"{file_path}:{yaml_path}"
        context = []
        
        # If node exists in graph
        if node_id in self.graph:
            # Get parent nodes
            parents = list(self.graph.predecessors(node_id))
            for parent in parents[:3]:  # Limit to avoid too much context
                parent_data = self.graph.nodes[parent]
                context.append(("parent", parent_data))
            
            # Get sibling nodes
            for parent in parents:
                siblings = list(self.graph.successors(parent))
                for sibling in siblings[:5]:  # Limit siblings
                    if sibling != node_id:
                        sibling_data = self.graph.nodes[sibling]
                        context.append(("sibling", sibling_data))
            
            # Get similar nodes
            similar_nodes = []
            for node, attrs in self.graph.nodes(data=True):
                if attrs.get('key') == self.graph.nodes[node_id].get('key') and node != node_id:
                    similar_nodes.append(node)
            
            for similar in similar_nodes[:3]:  # Limit similar nodes
                similar_data = self.graph.nodes[similar]
                context.append(("similar", similar_data))
        
        return context

# 3. YAML Structure Analyzer
class YAMLStructureAnalyzer:
    def __init__(self):
        self.indent_level = 2  # Default YAML indent level
        
    def analyze_indentation(self, content: str) -> int:
        """Analyze YAML indentation level."""
        lines = content.split("\n")
        indent_counts = {}
        
        for line in lines:
            if line.strip() and not line.strip().startswith("#"):
                leading_spaces = len(line) - len(line.lstrip(" "))
                if leading_spaces > 0:
                    indent_counts[leading_spaces] = indent_counts.get(leading_spaces, 0) + 1
        
        if indent_counts:
            # Find the most common indent level
            self.indent_level = min(indent_counts, key=lambda x: (-indent_counts[x], x))
        
        return self.indent_level
    
    def get_current_yaml_path(self, content: str, line_number: int) -> str:
        """Extract the YAML path for the current line."""
        lines = content.split("\n")[:line_number]
        path = []
        current_indent = 0
        
        for i, line in enumerate(lines):
            if not line.strip() or line.strip().startswith("#"):
                continue
                
            leading_spaces = len(line) - len(line.lstrip(" "))
            key = line.lstrip(" ").split(":")[0].strip()
            
            if leading_spaces == 0:  # Top level
                path = [key]
                current_indent = 0
            elif leading_spaces > current_indent:  # Child
                path.append(key)
                current_indent = leading_spaces
            elif leading_spaces < current_indent:  # Back to parent level
                # Pop until we find the right level
                levels_to_pop = (current_indent - leading_spaces) // self.indent_level
                path = path[:-levels_to_pop]
                path[-1] = key
                current_indent = leading_spaces
            elif leading_spaces == current_indent:  # Same level
                path[-1] = key
        
        return ".".join(path)
    
    def get_completion_suggestions(self, file_path: str, content: str, line_number: int, column: int) -> List[str]:
        """Get structural suggestions based on YAML indentation."""
        lines = content.split("\n")
        current_line = lines[line_number - 1] if line_number <= len(lines) else ""
        current_indent = len(current_line) - len(current_line.lstrip(" "))
        
        # Get the section key
        yaml_path = self.get_current_yaml_path(content, line_number - 1)
        
        # Common TOSCA structures based on indentation level
        suggestions = []
        
        # Top level structures
        if current_indent == 0:
            suggestions = [
                "tosca_definitions_version: ",
                "node_types:",
                "relationship_types:",
                "capability_types:",
                "artifact_types:",
                "data_types:",
                "interface_types:",
                "policy_types:",
                "topology_template:"
            ]
        # Second level structures
        elif current_indent == self.indent_level:
            if "node_types" in yaml_path:
                suggestions = [
                    "derived_from: ",
                    "version: ",
                    "metadata:",
                    "description: ",
                    "properties:",
                    "attributes:",
                    "capabilities:",
                    "requirements:",
                    "interfaces:"
                ]
            elif "topology_template" in yaml_path:
                suggestions = [
                    "description: ",
                    "inputs:",
                    "node_templates:",
                    "relationship_templates:",
                    "outputs:",
                    "groups:",
                    "policies:"
                ]
        # Third level structures
        elif current_indent == self.indent_level * 2:
            if "properties" in yaml_path:
                suggestions = [
                    "type: ",
                    "description: ",
                    "required: ",
                    "default: ",
                    "constraints:"
                ]
            elif "node_templates" in yaml_path:
                suggestions = [
                    "type: ",
                    "properties:",
                    "requirements:",
                    "capabilities:",
                    "interfaces:",
                    "artifacts:"
                ]
        
        return suggestions

# 4. MCP Server Implementation
class MCPServer:
    def __init__(self):
        self.semantic_rag = SemanticRAG()
        self.graph_rag = GraphRAG()
        self.yaml_analyzer = YAMLStructureAnalyzer()
        self.llm = OpenAI(model_name=LLM_MODEL)
        
        # Initialize prompt template for code completion
        self.completion_prompt = PromptTemplate(
            input_variables=["prefix", "suffix", "similar_chunks", "structural_context", "graph_context"],
            template="""
            You are a code completion assistant for a TOSCA-based infrastructure automation tool.
            Given the context, provide the best completion suggestions for the YAML file.
            
            Current context:
            ```yaml
            {prefix}█{suffix}
            ```
            
            Similar chunks from other files:
            {similar_chunks}
            
            Structural context:
            {structural_context}
            
            Graph relationships:
            {graph_context}
            
            Provide 3-5 completion suggestions as a JSON list with the following format:
            ```json
            [
              {{"text": "completion text", "description": "Brief description"}}
            ]
            ```
            
            Only include the completion text that should be inserted at the cursor position (marked by █).
            Make sure your suggestions are valid YAML and follow TOSCA conventions.
            """
        )
        
        self.completion_chain = LLMChain(
            llm=self.llm,
            prompt=self.completion_prompt
        )
    
    def initialize(self, folder_path: str) -> None:
        """Initialize all components with the provided folder."""
        self.semantic_rag.index_documents(folder_path)
        self.graph_rag.parse_yaml_to_graph(folder_path)
        print("MCP Server initialized successfully")
    
    def get_completion(self, file_context: FileContext) -> CompletionResponse:
        """Get completion suggestions based on multiple contexts."""
        # Analyze YAML structure
        indent_level = self.yaml_analyzer.analyze_indentation(file_context.content)
        yaml_path = self.yaml_analyzer.get_current_yaml_path(
            file_context.content, 
            file_context.line_number
        )
        
        # Get structural suggestions
        structural_suggestions = self.yaml_analyzer.get_completion_suggestions(
            file_context.file_path,
            file_context.content,
            file_context.line_number,
            file_context.column_number
        )
        
        # Get semantic suggestions from RAG
        similar_chunks = self.semantic_rag.retrieve_similar(
            file_context.prefix[-300:],  # Use the last 300 chars as context
            k=3
        )
        
        # Get graph context
        graph_context = self.graph_rag.get_structural_context(
            file_context.file_path,
            yaml_path
        )
        
        # Prepare context for LLM
        similar_chunks_text = "\n".join([
            f"Chunk {i+1} from {doc.metadata.get('source', 'unknown')}:\n```yaml\n{doc.page_content}\n```"
            for i, doc in enumerate(similar_chunks)
        ])
        
        structural_context_text = "\n".join([
            f"- {suggestion}" for suggestion in structural_suggestions
        ])
        
        graph_context_text = "\n".join([
            f"- {relation} node: {data.get('key')} (from {data.get('file_path')})"
            for relation, data in graph_context
        ])
        
        # Get LLM suggestions
        llm_result = self.completion_chain.run(
            prefix=file_context.prefix,
            suffix=file_context.suffix,
            similar_chunks=similar_chunks_text,
            structural_context=structural_context_text,
            graph_context=graph_context_text
        )
        
        try:
            # Parse LLM suggestions
            # Extract the JSON part
            json_start = llm_result.find("[")
            json_end = llm_result.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = llm_result[json_start:json_end]
                llm_suggestions = json.loads(json_str)
            else:
                # Fallback if JSON parsing fails
                llm_suggestions = [{"text": s, "description": "LLM suggestion"} for s in llm_result.split("\n") if s.strip()]
        except Exception as e:
            print(f"Error parsing LLM suggestions: {e}")
            llm_suggestions = []
        
        # Combine all suggestions
        candidates = []
        
        # Add structural suggestions (high priority)
        for suggestion in structural_suggestions:
            candidates.append(
                CompletionCandidate(
                    text=suggestion,
                    score=0.9,
                    source="structural"
                )
            )
        
        # Add LLM suggestions
        for suggestion in llm_suggestions:
            candidates.append(
                CompletionCandidate(
                    text=suggestion["text"],
                    score=0.8,
                    source="llm"
                )
            )
        
        # Sort by score
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        return CompletionResponse(
            candidates=candidates[:10],  # Limit to top 10
            context_used={
                "yaml_path": yaml_path,
                "similar_chunks": [doc.metadata.get("source", "") for doc in similar_chunks],
                "graph_context": [f"{relation}:{data.get('key', '')}" for relation, data in graph_context]
            }
        )

# API endpoints
@app.post("/initialize")
async def initialize_server(request: Request):
    data = await request.json()
    folder_path = data.get("folder_path")
    
    if not folder_path or not os.path.exists(folder_path):
        raise HTTPException(status_code=400, detail="Invalid folder path")
    
    server = MCPServer()
    server.initialize(folder_path)
    
    return {"status": "success", "message": "Server initialized successfully"}

@app.post("/complete")
async def get_completion(request: Request):
    data = await request.json()
    
    file_context = FileContext(
        file_path=data.get("file_path", ""),
        content=data.get("content", ""),
        line_number=data.get("line_number", 1),
        column_number=data.get("column_number", 1),
        prefix=data.get("prefix", ""),
        suffix=data.get("suffix", "")
    )
    
    server = MCPServer()
    completion_response = server.get_completion(file_context)
    
    return {
        "candidates": [
            {"text": c.text, "score": c.score, "source": c.source} 
            for c in completion_response.candidates
        ],
        "context_used": completion_response.context_used
    }

# VSCode extension integration helper
@app.post("/vscode-complete")
async def vscode_complete(request: Request):
    data = await request.json()
    
    file_context = FileContext(
        file_path=data.get("document", {}).get("uri", "").replace("file://", ""),
        content=data.get("document", {}).get("text", ""),
        line_number=data.get("position", {}).get("line", 1) + 1,  # VSCode is 0-indexed
        column_number=data.get("position", {}).get("character", 1) + 1,
        prefix=data.get("context", {}).get("prefix", ""),
        suffix=data.get("context", {}).get("suffix", "")
    )
    
    server = MCPServer()
    completion_response = server.get_completion(file_context)
    
    # Format response for VSCode
    return {
        "items": [
            {
                "label": c.text,
                "kind": 1,  # CompletionItemKind.Text
                "detail": f"Source: {c.source}",
                "insertText": c.text
            }
            for c in completion_response.candidates
        ]
    }

# Main entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TOSCA Code Completion Server')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    parser.add_argument('--host', type=str, default="127.0.0.1", help='Server host')
    parser.add_argument('--init-folder', type=str, help='Initialize with folder')
    
    args = parser.parse_args()
    
    if args.init_folder:
        server = MCPServer()
        server.initialize(args.init_folder)
    
    uvicorn.run(app, host=args.host, port=args.port)
