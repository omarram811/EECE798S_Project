from fastapi import APIRouter, Request, Depends, HTTPException, Form
from fastapi.responses import StreamingResponse, RedirectResponse
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.orm import Session
from typing import List, Dict
from .db import get_db
from .models import Agent, Conversation, Message
from .security import get_current_user_id
from .rag import retrieve, provider_from
import json, time, threading

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
    context_text = "\n\n".join([f"[{i+1}] {d['metadata'].get('title','')}\n{d['text'][:1200]}" for i,d in enumerate(context_docs)])
    system = {"role":"system","content": f"You are a course TA agent. Persona: {agent.persona}\nUse the following context from the course Drive when helpful. Cite filenames in your answers when you use them.\nContext:\n{context_text}"}
    provider = provider_from(agent)
    def event_gen():
        try:
            stream = provider.stream_chat([system, *history])
            acc = ""
            for token in stream:
                acc += token
                yield {"event":"message", "data": token}
            # save assistant message
            if acc.strip():
                db.add(Message(conversation_id=conv.id, user_id=uid, role="assistant", content=acc))
                db.commit()
            yield {"event":"done", "data":"[END]"}
        except Exception as e:
            yield {"event":"error", "data": str(e)}
        finally:
            with _STREAMS_LOCK:
                _ACTIVE_STREAMS.discard(conv.id)
    return EventSourceResponse(event_gen())
