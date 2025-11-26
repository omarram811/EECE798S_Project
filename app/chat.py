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
from .rag import load_drive_images, _extract_text_for_images
import uuid

def save_drive_images_to_db(agent, db: Session):
    """
    Loads images from Google Drive, performs OCR, and saves them to AgentFile table if not already present.
    """
    images = load_drive_images(agent)
    _extract_text_for_images(images, agent.owner_id)

    for img in images:
        if db.query(AgentFile).filter_by(agent_id=agent.id, title=img["title"]).first():
            continue

        db.add(AgentFile(
            agent_id=agent.id,
            file_id=str(uuid.uuid4()),
            title=img["title"],
            page=None,
            text=img["text"],
            last_modified=img["last_modified"]
        ))

    db.commit()


router = APIRouter(prefix="/chat", tags=["chat"])
_ACTIVE_STREAMS: set[int] = set()
_CANCELLED_STREAMS: set[int] = set()  # Track cancelled conversations
_STREAMS_LOCK = threading.Lock()



def _ensure_conversation(db, agent_id: int, user_id: int) -> Conversation:
    c = db.query(Conversation).filter_by(agent_id=agent_id, user_id=user_id).order_by(Conversation.id.desc()).first()
    if not c:
        c = Conversation(agent_id=agent_id, user_id=user_id, title="Conversation",summary="")
        db.add(c); db.commit()
    return c

def _get_last_messages(db, conv: Conversation, k: int = 4):
    msgs = (
        db.query(Message)
          .filter_by(conversation_id=conv.id)
          .order_by(Message.id.asc())
          .all()
    )

    active_msgs = [m for m in msgs if not getattr(m, "is_summarized", False)]

    return active_msgs[-k:], msgs

def _summarize_memory(provider, summary: str, old_messages: List[Message]):
    print("\n===== RUNNING SUMMARIZER =====")
    print(f"Old messages: {len(old_messages)}")
    print( " summary" , summary)
    print("================================\n")

    text = ""

    if summary:
        text += f"Previous summary:\n{summary}\n\n"

    for m in old_messages:
        text += f"{m.role.upper()}: {m.content}\n"

    prompt = [
        {
            "role": "system",
            "content": (
            "Summarize the following conversation context into at most 6 concise sentences. "
            "Preserve all important facts, tasks, questions, and decisions. "
            "Do NOT include trivial greetings or stylistic fluff. "
            "Make the summary factual, compressed, and directly usable as conversation memory."
            ),
        },
        {"role": "user", "content": text},
    ]

    print(f">>> SUMMARY PROMPT LENGTH = {len(text)} chars")

    try:
        result = provider.complete(prompt)
        print(f">>> RAW SUMMARY RESPONSE:\n{result}")
    except Exception as e:
        print(f"[SUMMARY ERROR] {e}")
        return summary or ""

    if not result:
        return summary or ""

    return result.strip()



@router.post("/{agent_id}")
def post_message(agent_id: int, request: Request, prompt: str = Form(...), db: Session = Depends(get_db)):
    uid = get_current_user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="Login required")
    agent = db.get(Agent, agent_id)
    if not agent:
        raise HTTPException(status_code=404)
    prompt = (prompt or "").strip()
    if not prompt:
        return RedirectResponse(f"/a/{agent.slug}", status_code=302)
    conv = _ensure_conversation(db, agent.id, uid)
    db.add(Message(conversation_id=conv.id, user_id=uid, role="user", content=prompt)); db.commit()
    return RedirectResponse(f"/a/{agent.slug}", status_code=302)

@router.post("/{agent_id}/cancel")
def cancel_stream(agent_id: int, request: Request, db: Session = Depends(get_db)):
    """Cancel the current stream for this agent and user"""
    uid = get_current_user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="Login required")
    agent = db.get(Agent, agent_id)
    if not agent:
        raise HTTPException(status_code=404)
    
    conv = _ensure_conversation(db, agent.id, uid)
    with _STREAMS_LOCK:
        _CANCELLED_STREAMS.add(conv.id)
    
    return {"status": "cancelled"}

@router.post("/public/{agent_id}/cancel")
def cancel_public_stream(agent_id: int, request: Request, db: Session = Depends(get_db)):
    """Cancel the current stream for public chat"""
    agent = db.get(Agent, agent_id)
    if not agent:
        raise HTTPException(status_code=404)
    
    session_id = request.session.get("public_session_id")
    if not session_id:
        return {"status": "no_session"}
    
    public_user_id = -abs(hash(session_id)) % (10 ** 8)
    conv = _ensure_conversation(db, agent.id, public_user_id)
    with _STREAMS_LOCK:
        _CANCELLED_STREAMS.add(conv.id)
    
    return {"status": "cancelled"}

@router.post("/public/{agent_id}")
def post_public_message(agent_id: int, request: Request, prompt: str = Form(...), db: Session = Depends(get_db)):
    agent = db.get(Agent, agent_id)
    if not agent:
        raise HTTPException(status_code=404)
    
    session_id = request.session.get("public_session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session["public_session_id"] = session_id
    
    public_user_id = -abs(hash(session_id)) % (10 ** 8)
    
    prompt = (prompt or "").strip()
    if not prompt:
        return RedirectResponse(f"/public/{agent.slug}", status_code=302)
    
    conv = _ensure_conversation(db, agent.id, public_user_id)
    db.add(Message(conversation_id=conv.id, user_id=public_user_id, role="user", content=prompt)); db.commit()
    return RedirectResponse(f"/public/{agent.slug}", status_code=302)


@router.get("/debug/conversations")
def debug_conversations(db: Session = Depends(get_db)):
    conversations = db.query(Conversation).all()
    return [
        {
            "id": c.id,
            "agent_id": c.agent_id,
            "user_id": c.user_id,
            "title": c.title,
            "summary": c.summary,
        }
        for c in conversations
    ]


@router.get("/{agent_id}/stream")
def stream(agent_id: int, request: Request, db: Session = Depends(get_db)):

    print("\n==============================")
    print("ENTERED /chat/<id>/stream")
    print("==============================")

    uid = get_current_user_id(request)
    print(f"[DEBUG] current_user_id = {uid}")

    if not uid:
        print("[ERROR] No user — returning auth error")
        return EventSourceResponse(({"event": "error", "data": "auth"} for _ in []))

    agent = db.get(Agent, agent_id)
    print(f"[DEBUG] agent loaded = {agent is not None}")

    if not agent:
        print("[ERROR] Agent not found.")
        return EventSourceResponse(({"event": "error", "data": "notfound"} for _ in []))

    conv = _ensure_conversation(db, agent.id, uid)
    print(f"[DEBUG] Conversation ID = {conv.id}")
    print(f"[DEBUG] Existing summary = '{conv.summary}'")

    last12, msgs = _get_last_messages(db, conv, k=12)
    print(f"[DEBUG] Last12 count = {len(last12)}")
    print(f"[DEBUG] Total msgs count = {len(msgs)}")


    history = [
        {"role": m.role, "content": m.content}
        for m in last12
        if m.role in ("user", "assistant") and (m.content or "").strip()
    ]
    print(f"[DEBUG] History entries = {len(history)}")


    if conv.summary and conv.summary.strip():
        print("[DEBUG] Injecting summary into history")
        history = [
            {"role": "system", "content": f"Conversation summary:\n{conv.summary}"}
        ] + history
    else:
        print("[DEBUG] No summary to inject")


    last = next((m for m in reversed(msgs) if (m.content or "").strip()), None)
    print(f"[DEBUG] Last message role = {last.role if last else None}")

    if not last or last.role != "user":
        print("[DEBUG] No new user message — IDLE")
        def idle():
            yield {"event": "done", "data": "[IDLE]"}
        return EventSourceResponse(idle())

    last_user = last.content
    print(f"[DEBUG] Last user message = {last_user[:80]}")

    with _STREAMS_LOCK:
        if conv.id in _ACTIVE_STREAMS:
            print("[DEBUG] Stream already active — BUSY")
            return EventSourceResponse(({"event": "info", "data": "[BUSY]"} for _ in []))
        _ACTIVE_STREAMS.add(conv.id)


    print("[DEBUG] Running RAG retrieval")
    context_docs = retrieve(agent, last_user, k=100)
    print(f"[DEBUG] Retrieved {len(context_docs)} context docs")

    context_text = "\n\n".join(
        [f"[{i+1}] {d['metadata'].get('title','')}\n{d['text'][:200]}"
         for i, d in enumerate(context_docs)]
    )

    provider = provider_from(agent)
    print("[DEBUG] Provider loaded OK")

    MAX_CONTEXT = 12
    SUMMARY_CHUNK = 8
    print(f"[DEBUG] Memory thresholds: MAX_CONTEXT={MAX_CONTEXT}, SUMMARY_CHUNK={SUMMARY_CHUNK}")

    def event_gen():
        print("STREAMING STARTED")

        try:
            yield {"event": "start", "data": ""}

            system = {
                "role": "system",
                "content": (
                    f"You are a course TA agent.\n"
                    f"Persona: {agent.persona}\n\n"
                    f"When you use a document, cite it by its exact filename.\n\n"
                    f"Retrieved context (may be partial):\n{context_text}"
                )
            }

            print("[DEBUG] Starting model.stream_chat call")
            stream = provider.stream_chat([system, *history])

            acc = ""
            for token in stream:
                # Cancellation check
                with _STREAMS_LOCK:
                    if conv.id in _CANCELLED_STREAMS:
                        print("Stream cancelled during output!")
                        _CANCELLED_STREAMS.discard(conv.id)
                        yield {"event": "cancelled", "data": "[CANCELLED]"}
                        return

                acc += token
                yield {"event": "message", "data": token}

            print(f"[DEBUG] Finished streaming. Accumulated {len(acc)} chars")

            
            if acc.strip():
                print("[DEBUG] Saving assistant message to DB")
                db.add(Message(
                    conversation_id=conv.id,
                    user_id=uid,
                    role="assistant",
                    content=acc,
                    is_summarized=False
                ))
                db.commit()

        
            _, all_msgs = _get_last_messages(db, conv, k=MAX_CONTEXT)
            active_msgs = [m for m in all_msgs if not m.is_summarized]
            print(f"[DEBUG] Active unsummarized msgs = {len(active_msgs)}")

            if len(active_msgs) > MAX_CONTEXT:
                print("TRIGGERING SUMMARIZATION")
                to_summarize = active_msgs[:-SUMMARY_CHUNK]
                print(f"[DEBUG] Summarizing {len(to_summarize)} msgs")
                print(f"[DEBUG] Previous summary = {conv.summary!r}")

                new_summary = _summarize_memory(provider, conv.summary, to_summarize)
                print(f"[DEBUG] New summary generated = {new_summary!r}")
                conv_persistent = db.merge(conv)   
                conv_persistent.summary = new_summary
                db.commit()
                db.refresh(conv_persistent)


                for m in to_summarize:
                    m.is_summarized = True

                db.commit()
                db.refresh(conv)
                print(f"[DEBUG] Summary saved. Length = {len(conv.summary)}")

            else:
                print("[DEBUG] Threshold not reached — skipping summarization")

            # Save logs
            db.add(QueryLog(agent_id=agent.id, query=last_user, response=acc))
            db.commit()
            print("[DEBUG] Query logged")

            yield {"event": "done", "data": "[END]"}

        except Exception as e:
            print(f"ERROR DURING STREAM: {e}")
            yield {"event": "error", "data": str(e)}

        finally:
            print("STREAM FINISHED — cleaning locks")
            with _STREAMS_LOCK:
                _ACTIVE_STREAMS.discard(conv.id)
                _CANCELLED_STREAMS.discard(conv.id)

    return EventSourceResponse(event_gen())


@router.get("/public/{agent_id}/stream")
def public_stream(agent_id: int, request: Request, db: Session = Depends(get_db)):
    agent = db.get(Agent, agent_id)
    if not agent:
        return EventSourceResponse(({"event": "error", "data": "notfound"} for _ in []))
    
    # Get session-based user ID
    session_id = request.session.get("public_session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session["public_session_id"] = session_id
    
    public_user_id = -abs(hash(session_id)) % (10 ** 8)
    
    conv = _ensure_conversation(db, agent.id, public_user_id)
    msgs = db.query(Message).filter_by(conversation_id=conv.id).order_by(Message.id.asc()).all()
    history = [
        {"role": m.role, "content": m.content}
        for m in msgs
        if m.role in ("user", "assistant") and (m.content or "").strip()
    ]
    last = next((m for m in reversed(msgs) if (m.content or "").strip()), None)
    if not last or last.role != "user":
        def idle():
            yield {"event": "done", "data": "[IDLE]"}
        return EventSourceResponse(idle())

    last_user = last.content

    with _STREAMS_LOCK:
        if conv.id in _ACTIVE_STREAMS:
            return EventSourceResponse(({"event": "info", "data": "[BUSY]"} for _ in []))
        _ACTIVE_STREAMS.add(conv.id)

    files = db.query(AgentFile).filter_by(agent_id=agent.id).all()
    file_list_text = "\n".join(
        [f"- {f.title}" for f in files]
    ) if files else "(No files found)"

    context_docs = retrieve(agent, last_user, k=100)
    context_text = "\n\n".join([f"[{i+1}] {d['metadata'].get('title','')}\n{d['text'][:1200]}" for i,d in enumerate(context_docs)])
    system = {
    "role": "system",
    "content": (
        f"You are a course TA agent.\n"
        f"Persona: {agent.persona}\n\n"
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
                # Check if stream was cancelled
                with _STREAMS_LOCK:
                    if conv.id in _CANCELLED_STREAMS:
                        _CANCELLED_STREAMS.discard(conv.id)
                        yield {"event":"cancelled", "data":"[CANCELLED]"}
                        return
                acc += token
                yield {"event":"message", "data": token}
                time.sleep(0.1)
            
            # Check again before saving to prevent race condition
            with _STREAMS_LOCK:
                if conv.id in _CANCELLED_STREAMS:
                    _CANCELLED_STREAMS.discard(conv.id)
                    yield {"event":"cancelled", "data":"[CANCELLED]"}
                    return
            
            if acc.strip():
                db.add(Message(conversation_id=conv.id, user_id=public_user_id, role="assistant", content=acc))
                db.commit()
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
                _CANCELLED_STREAMS.discard(conv.id)
    return EventSourceResponse(event_gen())
