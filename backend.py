from fastapi import FastAPI, UploadFile, File
import os
from dotenv import load_dotenv
from Processing import parse_pdf
from contextlib import asynccontextmanager
from llama_index.core.node_parser import SentenceSplitter
from Processing import process_images, describe_image
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from langchain_openrouter import ChatOpenRouter
from llama_index.core.indices import VectorStoreIndex
from langchain_core.messages import SystemMessage, HumanMessage
import chromadb
import asyncio
from fastapi import HTTPException
from pydantic import BaseModel
from Processing import system_prompt
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import base64
import uuid, json
load_dotenv()



@asynccontextmanager
async def lifespan(app: FastAPI):

    # Startup
    print("Loading RAG system...")

    app.state.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    app.state.llm = ChatOpenRouter(model="google/gemini-2.5-flash-lite", api_key="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", temperature=0.0)
    app.state.reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L6-v2", top_n=3)
    # App startup
    chroma_client = chromadb.PersistentClient(path='./chroma')
    app.state.chroma_collection = chroma_client.get_or_create_collection(name='multi_model_rag', metadata={'name':'chroma', "type":"persist"})
    app.state.vector_store = ChromaVectorStore(chroma_collection=app.state.chroma_collection)
    app.state.storage_context = StorageContext.from_defaults(
        vector_store=app.state.vector_store
    )

    print("RAG system loaded")
    yield
    # Shutdown
    print("Shutting down RAG system...")


app = FastAPI(title="Multi-RAG API", description="API for Multi-RAG", version="1.0.0", lifespan=lifespan)

#CORS
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.mount("/images", StaticFiles(directory="extracted_images"), name="images")

@app.get("/")
async def root():
    return {"message": "Welcome to the Multi-RAG API"}


@app.get("/health")
async def health():
    return {"status": "ok"}



UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
DOCUMENTS_FILE = "history.json.json"

# ----------------------------- # JSON file initialize # ----------------------------- 
def initialize_documents_file():
    if not os.path.exists(DOCUMENTS_FILE):
         with open(DOCUMENTS_FILE, "w", encoding="utf-8") as f:
             json.dump([], f, indent=4) 

initialize_documents_file()

# ----------------------------- # Read documents # ----------------------------- 
def read_documents(): 
    with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f: 
        return json.load(f)


# ----------------------------- # Save documents # ----------------------------- 
def save_documents(documents):
    with open(DOCUMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=4)


@app.post('/document/upload_file')
async def upload_document(
    file: UploadFile = File(...)
):
    # 1. Save file
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    content = await file.read()

    with open(file_path, "wb") as f:
        f.write(content)

    # 2. Parse PDF
    documents = await asyncio.to_thread(
        parse_pdf,
        file_path
    )

    # 3. Text chunks
    splitter = SentenceSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    text_chunks = splitter.get_nodes_from_documents(
        documents["text"]
    )

    # 4. Image processing
    # VLM calls ko parallel karo
    image_docs = await process_images(documents)

    # 5. Tables
    table_docs = documents["tables"]

    # 6. All nodes
    nodes = (
        text_chunks
        + table_docs
        + image_docs
    )

    # 7. Existing index use karo
    if not hasattr(app.state, "index"):
        app.state.index = VectorStoreIndex(
            nodes=nodes,
            storage_context=app.state.storage_context,
            embed_model=app.state.embed_model
        )
    else:
        app.state.index.insert_nodes(nodes)
    
    #manage history
    history = read_documents()
    document = { "id": str(uuid.uuid4()), "filename": file.filename, "status": "indexed" } 
    history.append(document) 
    save_documents(history)

    return {
        "message": "Document uploaded successfully",
        "chunks": len(text_chunks),
        "images": len(image_docs),
        "tables": len(table_docs)
    }



class QueryRequest(BaseModel):
    query: str


def encode_image(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")


@app.post("/ask")
async def ask_query(request: QueryRequest):

    query = request.query

    # Check index
    if not hasattr(app.state, "index"):
        raise HTTPException(
            status_code=400,
            detail="No document has been indexed yet"
        )

    index = app.state.index

    # -----------------------------
    # Retrieve
    # -----------------------------
    retriever = index.as_retriever(
        similarity_top_k=5,
        image_similarity_top_k=4
    )

    results = await asyncio.to_thread(
        retriever.retrieve,
        query
    )

    # -----------------------------
    # Rerank
    # -----------------------------
    reranked_results = await asyncio.to_thread(
        app.state.reranker.postprocess_nodes,
        results,
        query_str=query
    )

    # -----------------------------
    # Prepare context
    # -----------------------------
    text_context = []
    image_paths = []

    for node in reranked_results:

        text_context.append(
            node.get_content()
        )

        if node.metadata.get("category") == "Image":

            image_path = getattr(
                node.node,
                "image_path",
                None
            )

            if image_path:
                image_paths.append(image_path)

    context = "\n\n".join(text_context)

    # -----------------------------
    # LLM Content
    # -----------------------------
    content = [
        {
            "type": "text",
            "text": f"""
            Context:
            {context}

            Question:
            {query}
            """
        }
    ]

    # Add images
    for image_path in image_paths:

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        "data:image/jpeg;base64,"f"{encode_image(image_path)}"
                    )
                }
            }
        )


    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=content)
    ]

    # LLM call
    response = await asyncio.to_thread(
        app.state.llm.invoke,
        messages
    )

    # -----------------------------
    # Response
    # -----------------------------
    return {
        "question": query,
        "answer": response.content,
        "images": image_paths
    }





@app.get("/documents")
async def get_documents():
    documents = read_documents() 
    return { "count": len(documents), "documents": documents}

