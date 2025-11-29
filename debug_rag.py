# debug_rag.py (run from project root)
from app.db import SessionLocal
from app.models import Agent
from app.rag import reindex_agent, get_vector_client, ensure_collection, load_drive_docs

db = SessionLocal()
a = db.get(Agent, 1)  # adjust if needed
print("Agent:", a.name, a.drive_folder_id, a.embed_model, a.provider)

docs = load_drive_docs(a)
print("Docs loaded:", len(docs))
print("Sample titles:", [d["title"] for d in docs[:5]])

added, updated, total = reindex_agent(a, "")
print("Upserted:", added, updated, total)

client = get_vector_client()
col = ensure_collection(a, client)
print("Collection count:", col.count())

out = col.query(query_texts=["types of prompting"], n_results=3)
for i, (doc, md) in enumerate(zip(out.get("documents",[[]])[0], out.get("metadatas",[[]])[0])):
    print(f"\nHit {i+1}: {md.get('title')}  page={md.get('page')} slide={md.get('slide')}")
    print(doc[:300], "…")
