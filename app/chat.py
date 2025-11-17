# from fastapi import APIRouter, Request, Depends, HTTPException, Form
# from fastapi.responses import StreamingResponse, RedirectResponse
# from sse_starlette.sse import EventSourceResponse
# from sqlalchemy.orm import Session
# from typing import List, Dict
# from .db import get_db
# from .models import Agent, Conversation, Message
# from .security import get_current_user_id
# from .rag import retrieve, provider_from
# import json, time, threading

# router = APIRouter(prefix="/chat", tags=["chat"])
# _ACTIVE_STREAMS: set[int] = set()
# _STREAMS_LOCK = threading.Lock()

# def _ensure_conversation(db, agent_id: int, user_id: int) -> Conversation:
#     c = db.query(Conversation).filter_by(agent_id=agent_id, user_id=user_id).order_by(Conversation.id.desc()).first()
#     if not c:
#         c = Conversation(agent_id=agent_id, user_id=user_id, title="Conversation")
#         db.add(c); db.commit()
#     return c

# @router.post("/{agent_id}")
# def post_message(agent_id: int, request: Request, prompt: str = Form(...), db: Session = Depends(get_db)):
#     uid = get_current_user_id(request)
#     if not uid:
#         raise HTTPException(status_code=401, detail="Login required")
#     agent = db.get(Agent, agent_id)
#     if not agent:
#         raise HTTPException(status_code=404)
#     # Trim and ignore empty messages to avoid empty 'content' later
#     prompt = (prompt or "").strip()
#     if not prompt:
#         return RedirectResponse(f"/a/{agent.slug}", status_code=302)
#     conv = _ensure_conversation(db, agent.id, uid)
#     db.add(Message(conversation_id=conv.id, user_id=uid, role="user", content=prompt)); db.commit()
#     return RedirectResponse(f"/a/{agent.slug}", status_code=302)

# @router.get("/{agent_id}/stream")
# def stream(agent_id: int, request: Request, db: Session = Depends(get_db)):
#     uid = get_current_user_id(request)
#     if not uid:
#         return EventSourceResponse(({"event": "error", "data": "auth"} for _ in []))
#     agent = db.get(Agent, agent_id)
#     if not agent:
#         return EventSourceResponse(({"event": "error", "data": "notfound"} for _ in []))
#     conv = _ensure_conversation(db, agent.id, uid)
#     # build context (skip empty contents entirely)
#     msgs = db.query(Message).filter_by(conversation_id=conv.id).order_by(Message.id.asc()).all()
#     history = [
#         {"role": m.role, "content": m.content}
#         for m in msgs
#         if m.role in ("user", "assistant") and (m.content or "").strip()
#     ]
#     # Stream ONLY if the last non-empty message is from the user
#     last = next((m for m in reversed(msgs) if (m.content or "").strip()), None)
#     if not last or last.role != "user":
#         def idle():
#             # Tell the client we're done with nothing to stream (no error UI)
#             yield {"event": "done", "data": "[IDLE]"}
#         return EventSourceResponse(idle())

#     last_user = last.content

#     # Prevent concurrent streams for the same conversation (debounce)
#     with _STREAMS_LOCK:
#         if conv.id in _ACTIVE_STREAMS:
#             return EventSourceResponse(({"event": "info", "data": "[BUSY]"} for _ in []))
#         _ACTIVE_STREAMS.add(conv.id)

#     # RAG retrieve only when we have a real query
#     context_docs = retrieve(agent, last_user, k=5)
#     context_text = "\n\n".join([f"[{i+1}] {d['metadata'].get('title','')}\n{d['text'][:1200]}" for i,d in enumerate(context_docs)])
#     system = {"role":"system","content": f"You are a course TA agent. Persona: {agent.persona}\nUse the following context from the course Drive when helpful. Cite filenames in your answers when you use them.\nContext:\n{context_text}"}
#     provider = provider_from(agent)
#     def event_gen():
#         try:
#             stream = provider.stream_chat([system, *history])
#             acc = ""
#             for token in stream:
#                 acc += token
#                 yield {"event":"message", "data": token}
#             # save assistant message
#             if acc.strip():
#                 db.add(Message(conversation_id=conv.id, user_id=uid, role="assistant", content=acc))
#                 db.commit()
#             yield {"event":"done", "data":"[END]"}
#         except Exception as e:
#             yield {"event":"error", "data": str(e)}
#         finally:
#             with _STREAMS_LOCK:
#                 _ACTIVE_STREAMS.discard(conv.id)
#     return EventSourceResponse(event_gen())



from fastapi import APIRouter, Request, Depends, HTTPException, Form
from fastapi.responses import StreamingResponse, RedirectResponse
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.orm import Session
from typing import List, Dict
from .db import get_db
from .models import Agent, Conversation, Message, AgentFile
from .security import get_current_user_id
from .rag import retrieve, provider_from
import json, time, threading
from .models import QueryLog

router = APIRouter(prefix="/chat", tags=["chat"])
_ACTIVE_STREAMS: set[int] = set()
_STREAMS_LOCK = threading.Lock()

def _ensure_conversation(db, agent_id: int, user_id: int) -> Conversation:
    c = db.query(Conversation).filter_by(agent_id=agent_id, user_id=user_id).order_by(Conversation.id.desc()).first()
    if not c:
        c = Conversation(agent_id=agent_id, user_id=user_id, title="Conversation")
        db.add(c); db.commit()
    return c

@router.post("/{agent_id}")
def post_message(agent_id: int, request: Request, prompt: str = Form(...), db: Session = Depends(get_db)):
    uid = get_current_user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="Login required")
    agent = db.get(Agent, agent_id)
    if not agent:
        raise HTTPException(status_code=404)
    # Trim and ignore empty messages to avoid empty 'content' later
    prompt = (prompt or "").strip()
    if not prompt:
        return RedirectResponse(f"/a/{agent.slug}", status_code=302)
    conv = _ensure_conversation(db, agent.id, uid)
    db.add(Message(conversation_id=conv.id, user_id=uid, role="user", content=prompt)); db.commit()
    return RedirectResponse(f"/a/{agent.slug}", status_code=302)

@router.get("/{agent_id}/stream")
def stream(agent_id: int, request: Request, db: Session = Depends(get_db)):
    uid = get_current_user_id(request)
    if not uid:
        return EventSourceResponse(({"event": "error", "data": "auth"} for _ in []))
    agent = db.get(Agent, agent_id)
    if not agent:
        return EventSourceResponse(({"event": "error", "data": "notfound"} for _ in []))

    # # ---- NEW: Fetch the true list of files from metadata ----
    files = (
        db.query(AgentFile)
        .filter_by(agent_id=agent.id)
        .order_by(AgentFile.title.asc())
        .all()
    )

    # Build a clean, deterministic list
    file_list_text = "\n".join(
        f"- {f.title}" + (f" (page {f.page})" if f.page else "") 
        for f in files
    )
    if not file_list_text:
        file_list_text = "(No files indexed yet)"


    conv = _ensure_conversation(db, agent.id, uid)
    # build context (skip empty contents entirely)
    msgs = db.query(Message).filter_by(conversation_id=conv.id).order_by(Message.id.asc()).all()
    history = [
        {"role": m.role, "content": m.content}
        for m in msgs
        if m.role in ("user", "assistant") and (m.content or "").strip()
    ]
    # Stream ONLY if the last non-empty message is from the user
    last = next((m for m in reversed(msgs) if (m.content or "").strip()), None)
    if not last or last.role != "user":
        def idle():
            # Tell the client we're done with nothing to stream (no error UI)
            yield {"event": "done", "data": "[IDLE]"}
        return EventSourceResponse(idle())

    last_user = last.content

    # Prevent concurrent streams for the same conversation (debounce)
    with _STREAMS_LOCK:
        if conv.id in _ACTIVE_STREAMS:
            return EventSourceResponse(({"event": "info", "data": "[BUSY]"} for _ in []))
        _ACTIVE_STREAMS.add(conv.id)

    # RAG retrieve only when we have a real query
    context_docs = retrieve(agent, last_user, k=5)
    context_text = "\n\n".join([f"[{i+1}] {d['metadata'].get('title','')}\n{d['text'][:300]}" for i,d in enumerate(context_docs)])
    system = {
    "role": "system",
    "content": (
        f"You are a course TA agent.\n"
        f"Persona: {agent.persona}\n\n"
         f"Below is the files you have access to. "
        f"{file_list_text}\n\n"
        f"When you use a document, cite it by its exact filename.\n\n"
        f"Retrieved context (may be partial):\n{context_text}"
    )
}

    provider = provider_from(agent)
    def event_gen():
        try:
            stream = provider.stream_chat([system, *history])
            acc = ""
            for token in stream:
                acc += token
                yield {"event":"message", "data": token}
                time.sleep(0.1) ##added here for streaming
            # save assistant message
            if acc.strip():
                db.add(Message(conversation_id=conv.id, user_id=uid, role="assistant", content=acc))
                db.commit()
                 # ------------------------------------------
                # LOG THE QUERY & RESPONSE
                # ------------------------------------------
                user_question = history[-1]["content"] if history else ""
                db.add(QueryLog(
                    agent_id=agent.id,
                    query=user_question,
                    response=acc
                ))
                db.commit()

            yield {"event":"done", "data":"[END]"}
        except Exception as e:
            yield {"event":"error", "data": str(e)}
        finally:
            with _STREAMS_LOCK:
                _ACTIVE_STREAMS.discard(conv.id)
    return EventSourceResponse(event_gen())



# from fastapi import APIRouter, Request, Depends, HTTPException, Form
# from fastapi.responses import RedirectResponse
# from sse_starlette.sse import EventSourceResponse
# from sqlalchemy.orm import Session
# from typing import List, Dict
# from .db import get_db
# from .models import Agent, Conversation, Message
# from .security import get_current_user_id
# from .rag import retrieve, provider_from

# import json, threading

# router = APIRouter(prefix="/chat", tags=["chat"])
# _ACTIVE_STREAMS: set[int] = set()
# _STREAMS_LOCK = threading.Lock()


# # ---------------------------------------------
# # Conversation helper
# # ---------------------------------------------
# def _ensure_conversation(db, agent_id: int, user_id: int) -> Conversation:
#     c = (
#         db.query(Conversation)
#         .filter_by(agent_id=agent_id, user_id=user_id)
#         .order_by(Conversation.id.desc())
#         .first()
#     )
#     if not c:
#         c = Conversation(agent_id=agent_id, user_id=user_id, title="Conversation")
#         db.add(c)
#         db.commit()
#     return c


# # ---------------------------------------------
# # POST /chat/{agent_id}
# # ---------------------------------------------
# @router.post("/{agent_id}")
# def post_message(agent_id: int, request: Request, prompt: str = Form(...), db: Session = Depends(get_db)):
#     uid = get_current_user_id(request)
#     if not uid:
#         raise HTTPException(status_code=401, detail="Login required")

#     agent = db.get(Agent, agent_id)
#     if not agent:
#         raise HTTPException(status_code=404)

#     prompt = (prompt or "").strip()
#     if not prompt:
#         return RedirectResponse(f"/a/{agent.slug}", status_code=302)

#     conv = _ensure_conversation(db, agent.id, uid)
#     db.add(Message(conversation_id=conv.id, user_id=uid, role="user", content=prompt))
#     db.commit()

#     return RedirectResponse(f"/a/{agent.slug}", status_code=302)


# # ---------------------------------------------
# # GET /chat/{agent_id}/stream
# # ---------------------------------------------
# @router.get("/{agent_id}/stream")
# def stream(agent_id: int, request: Request, db: Session = Depends(get_db)):
#     uid = get_current_user_id(request)
#     if not uid:
#         return EventSourceResponse(({"event": "error", "data": "auth"} for _ in []))

#     agent = db.get(Agent, agent_id)
#     if not agent:
#         return EventSourceResponse(({"event": "error", "data": "notfound"} for _ in []))

#     conv = _ensure_conversation(db, agent.id, uid)

#     # --------------------------
#     # Build chat history
#     # --------------------------
#     msgs = (
#         db.query(Message)
#         .filter_by(conversation_id=conv.id)
#         .order_by(Message.id.asc())
#         .all()
#     )

#     history = [
#         {"role": m.role, "content": m.content}
#         for m in msgs
#         if m.role in ("user", "assistant") and (m.content or "").strip()
#     ]

#     # Only stream if user sent something new
#     last = next((m for m in reversed(msgs) if (m.content or "").strip()), None)
#     if not last or last.role != "user":
#         return EventSourceResponse(
#             ({"event": "done", "data": "[IDLE]"} for _ in [])
#         )

#     last_user_query = last.content

#     # --------------------------
#     # Prevent concurrent streams
#     # --------------------------
#     with _STREAMS_LOCK:
#         if conv.id in _ACTIVE_STREAMS:
#             return EventSourceResponse(({"event": "info", "data": "[BUSY]"} for _ in []))
#         _ACTIVE_STREAMS.add(conv.id)

#     # --------------------------
#     # Retrieve **RAG** documents (already chunked)
#     # --------------------------
#     results = retrieve(agent, last_user_query, k=5)
#     # 1. Extract memory from the new user message
#     memories = extract_memory_from_message(provider, last_user_query)

#     # 2. Save memories
#     if memories:
#         store_memories(db, provider, uid, agent.id, memories)

#     # 3. Retrieve relevant memory for the current query
#     memory_hits = retrieve_relevant_memories(db, provider, uid, agent.id, last_user_query, k=5)

#     memory_text = "\n".join([f"- {m}" for m in memory_hits]) if memory_hits else "No long-term memory found."
#         # --------------------------
#     # Build **clean** context (NO 1200-char dumps)
#     # --------------------------
#     context_blocks = []
#     for i, d in enumerate(results):
#         title = d["metadata"].get("title", "") or "(Untitled)"
#         source = d["metadata"].get("source", "")
#         page = d["metadata"].get("page")
#         slide = d["metadata"].get("slide")

#         # Use short snippet (~200 chars)
#         snippet = (d["text"] or "").replace("\n", " ")[:200]

#         block = f"""[{i+1}] {title}
# Snippet: {snippet}...
# Source: {source}  page={page} slide={slide}"""

#         context_blocks.append(block)

#     context_text = "\n\n".join(context_blocks) if context_blocks else "No matching documents."

#     # --------------------------
#     # Build final system prompt
#     # --------------------------
#     system = {
#     "role": "system",
#     "content": (
#         f"You are a course TA agent.\n"
#         f"Persona: {agent.persona}\n\n"
#         f"===== LONG-TERM MEMORY =====\n"
#         f"{memory_text}\n\n"
#         f"===== RETRIEVED COURSE MATERIAL =====\n"
#         f"{context_text}\n\n"
#         f"Use long-term memory for personalization. "
#         f"Use course materials for correctness. "
#         f"Never invent citations."
#     ),
#     }


#     provider = provider_from(agent)


#     def event_gen():
#         try:
#             stream = provider.stream_chat([system_msg, *history])
#             acc = ""

#             for token in stream:
#                 acc += token
#                 yield {"event": "message", "data": token}

#             # Save assistant reply
#             if acc.strip():
#                 db.add(Message(conversation_id=conv.id, user_id=uid, role="assistant", content=acc))
#                 db.commit()

#             yield {"event": "done", "data": "[END]"}

#         except Exception as e:
#             yield {"event": "error", "data": str(e)}

#         finally:
#             with _LOCK:
#                 _ACTIVE_STREAMS.discard(conv.id)

#     return EventSourceResponse(event_gen())
