from fastapi import APIRouter, Request, Depends, Form, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from sqlalchemy.orm import Session
from .db import get_db
from .models import Agent, Conversation, GoogleToken, AgentFile
from .security import get_current_user_id
from .rag import reindex_agent
from .rag import provider_from
from .scheduler import _creds_path_for_user
from fastapi.templating import Jinja2Templates
from .models import QueryLog
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from .analytics import fetch_queries, cluster_queries, summarize_clusters
from .providers import validate_api_key_with_provider
import json
import tempfile
import uuid
from .progress import get_progress
import threading
from markdown import markdown

router = APIRouter(prefix="/agents", tags=["agents"])
templates = Jinja2Templates(directory="templates")


# ============ SUPPORTED MODELS (CURATED LIST) ============

SUPPORTED_MODELS = {
    "openai": {
        "chat": ["gpt-4.1-nano", "gpt-4o-mini", "gpt-5-mini", "gpt-5-nano", "gpt-5"],
        "embed": ["text-embedding-3-small", "text-embedding-ada-002"],
        "default_chat": "gpt-4o-mini",
        "default_embed": "text-embedding-3-small",
    },
    "gemini": {
        "chat": ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
        "embed": ["models/text-embedding-004", "models/gemini-embedding-001"],
        "default_chat": "gemini-2.5-flash",
        "default_embed": "models/text-embedding-004",
    },
}


def get_default_model(provider: str) -> str:
    """Get default chat model for a provider."""
    return SUPPORTED_MODELS.get(provider, SUPPORTED_MODELS["openai"])["default_chat"]


def get_default_embed_model(provider: str) -> str:
    """Get default embedding model for a provider."""
    return SUPPORTED_MODELS.get(provider, SUPPORTED_MODELS["openai"])["default_embed"]


def validate_api_key(provider: str, api_key: str) -> tuple[bool, str]:
    """Validate that API key format matches the provider's expected format."""
    if not api_key or not api_key.strip():
        return False, "API key is required."
    
    api_key = api_key.strip()
    
    if provider == "openai":
        if not api_key.startswith("sk-"):
            return False, "OpenAI API keys must start with 'sk-'. Please check your API key."
        if len(api_key) < 20:
            return False, "OpenAI API key seems too short. Please check your API key."
    elif provider == "gemini":
        if not api_key.startswith("AI"):
            return False, "Gemini API keys typically start with 'AI'. Please check your API key."
        if len(api_key) < 30:
            return False, "Gemini API key seems too short. Please check your API key."
    
    return True, ""


def validate_drive_folder(drive_folder: str) -> tuple[bool, str]:
    """Validate that a Google Drive folder is provided."""
    if not drive_folder or not drive_folder.strip():
        return False, "Google Drive folder URL or ID is required. Your TA agent needs course materials to learn from."
    return True, ""


def validate_agent_config(provider: str, api_key: str, drive_folder: str = None,
                          require_api_key: bool = True, require_drive: bool = True) -> tuple[bool, str]:
    """Validate agent configuration fields. Returns (is_valid, error_message)."""
    
    # Validate drive folder (required for creation)
    if require_drive and drive_folder is not None:
        valid, error = validate_drive_folder(drive_folder)
        if not valid:
            return False, error
    
    # Validate API key
    if require_api_key or (api_key and api_key.strip()):
        if api_key and api_key.strip():
            valid, error = validate_api_key(provider, api_key.strip())
            if not valid:
                return False, error
        elif require_api_key:
            return False, "API key is required."
    
    return True, ""


@router.post("/create")
def create_agent(request: Request,
                 name: str = Form(...),
                 drive_folder: str = Form(...),  # Required
                 persona: str = Form(""),
                 provider: str = Form("openai"),
                 model: str = Form(""),  # Optional, will use default
                 embed_model: str = Form(""),  # Optional, will use default
                 api_key: str = Form(...),
                 db: Session = Depends(get_db)):
    uid = get_current_user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="Login required")
    
    # Apply defaults for empty model fields
    final_model = model.strip() if model and model.strip() else get_default_model(provider)
    final_embed_model = embed_model.strip() if embed_model and embed_model.strip() else get_default_embed_model(provider)
    
    # Server-side validation of configuration (format check)
    is_valid, error_msg = validate_agent_config(
        provider, api_key, 
        drive_folder=drive_folder, require_api_key=True, require_drive=True
    )
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"{error_msg}")
    
    # ============ VALIDATE API KEY WITH PROVIDER ============
    # This makes a real API call to verify the key works before creating the agent
    api_valid, api_error = validate_api_key_with_provider(provider, api_key.strip(), final_embed_model)
    if not api_valid:
        raise HTTPException(status_code=400, detail=api_error)
    
    # Check if Google Drive is connected - required to create agents
    tok = db.query(GoogleToken).filter_by(user_id=uid).first()
    if not tok:
        raise HTTPException(
            status_code=400, 
            detail="Google Drive not connected. Please connect your Google Drive before creating an agent."
        )
    
    slug = str(uuid.uuid4())
    a = Agent(owner_id=uid, name=name, slug=slug, drive_folder_id=drive_folder, persona=persona,
              provider=provider, model=final_model, embed_model=final_embed_model, api_key=api_key)
    db.add(a); db.commit()
    
    # Start indexing in background thread (drive folder is required)
    agent_id = a.id  # Store agent ID for background thread
    creds_path = _creds_path_for_user(uid)
    
    # Run in background thread with its own DB session
    def background_index():
        from .db import SessionLocal
        from .progress import complete_progress
        db_bg = SessionLocal()
        try:
            # Fetch agent in background thread's session
            agent_bg = db_bg.get(Agent, agent_id)
            if agent_bg:
                # Enable progress tracking for initial agent creation
                # Force reindex since this is a new agent
                reindex_agent(agent_bg, creds_path, track_progress=True, force_reindex=True)
                print(f"[create_agent] Background indexing completed for agent {agent_id}")
            else:
                print(f"[create_agent] Agent {agent_id} not found in background thread")
                complete_progress(agent_id, "Agent not found during indexing")
        except Exception as e:
            error_msg = str(e)
            print(f"[create_agent] Background indexing failed for agent {agent_id}: {error_msg}")
            import traceback
            traceback.print_exc()
            # Mark progress as failed
            complete_progress(agent_id, error_msg)
        finally:
            db_bg.close()
    
    thread = threading.Thread(target=background_index, daemon=True)
    thread.start()
    
    # Redirect to progress page to show indexing progress
    return RedirectResponse(f"/agents/{a.id}/progress-page", status_code=302)


@router.post("/{agent_id}/delete")
def delete_agent(request: Request, agent_id: int, db: Session = Depends(get_db)):
    uid = get_current_user_id(request)
    agent = db.get(Agent, agent_id)

    if not agent or agent.owner_id != uid:
        raise HTTPException(status_code=404, detail="Agent not found")

    # 1. Delete conversations (cascade deletes messages)
    for conv in agent.conversations:
        db.delete(conv)

    # 2. Delete query logs
    for log in agent.query_logs:
        db.delete(log)

    # 3. AgentFiles will delete automatically via cascade now

    # 4. Delete vector embeddings
    try:
        from .rag import ensure_collection, get_vector_client
        client = get_vector_client()
        col = ensure_collection(agent, client)
        col.delete(where={"agent_id": agent_id})
    except Exception as e:
        print("[delete_agent] Warning:", e)

    # 5. Delete the agent itself
    db.delete(agent)
    db.commit()

    db.query(Agent).all()
    db.query(Conversation).all()
    db.query(QueryLog).all()
    db.query(AgentFile).all()

    return RedirectResponse("/dashboard", status_code=302)




@router.post("/{agent_id}/update")
def update_agent(request: Request, agent_id: int,
                 name: str = Form(...),
                 drive_folder: str = Form(""),
                 persona: str = Form(""),
                 announcement: str = Form(""),
                 provider: str = Form("openai"),
                 model: str = Form(""),  # Optional, will use default if empty
                 embed_model: str = Form(""),  # Optional, will use default if empty
                 api_key: str = Form(""),
                 db: Session = Depends(get_db)):
    uid = get_current_user_id(request)
    a = db.get(Agent, agent_id)
    if not a or a.owner_id != uid:
        raise HTTPException(status_code=404)
    
    # Apply defaults for empty model fields
    final_model = model.strip() if model and model.strip() else get_default_model(provider)
    final_embed_model = embed_model.strip() if embed_model and embed_model.strip() else get_default_embed_model(provider)
    
    # Server-side validation of configuration (format check)
    # Only validate API key format if a new one is provided (existing key is kept if empty)
    is_valid, error_msg = validate_agent_config(
        provider, api_key,
        drive_folder=drive_folder, require_api_key=False, require_drive=False
    )
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"{error_msg}")
    
    # ============ VALIDATE API KEY WITH PROVIDER ============
    # If API key is provided (new or changed), validate it with the provider
    # Also validate if provider or embed_model changed (since existing key needs to work with new settings)
    provider_changed = (a.provider != provider)
    embed_model_changed = (a.embed_model != final_embed_model)
    api_key_provided = bool(api_key and api_key.strip())
    
    # Determine which API key to validate
    key_to_validate = api_key.strip() if api_key_provided else a.api_key
    
    # Validate if: new key provided, OR provider changed, OR embed model changed
    if api_key_provided or provider_changed or embed_model_changed:
        if key_to_validate:
            api_valid, api_error = validate_api_key_with_provider(provider, key_to_validate, final_embed_model)
            if not api_valid:
                raise HTTPException(status_code=400, detail=api_error)
    
    # Check if embedding model or provider changed - this requires full re-embedding
    embedding_changed = (provider_changed or embed_model_changed)
    
    # Also check if drive folder changed - this requires re-indexing
    drive_folder_changed = (a.drive_folder_id != drive_folder)
    
    a.name, a.persona, a.drive_folder_id = name, persona, drive_folder
    a.announcement = announcement.strip() if announcement else None
    a.provider, a.model, a.embed_model = provider, final_model, final_embed_model
    if api_key:  # Only update API key if provided
        a.api_key = api_key
    db.commit()
    
    # If embedding model or drive folder changed, trigger re-indexing to rebuild vector store
    needs_reindex = (embedding_changed or drive_folder_changed) and drive_folder
    
    if needs_reindex:
        tok = db.query(GoogleToken).filter_by(user_id=uid).first()
        if tok:
            stored_agent_id = a.id  # Store agent ID for background thread
            creds_path = _creds_path_for_user(uid)
            
            # Run re-indexing in background thread with progress tracking
            def background_reindex():
                from .db import SessionLocal
                from .progress import complete_progress
                db_bg = SessionLocal()
                try:
                    # Fetch agent in background thread's session
                    agent_bg = db_bg.get(Agent, stored_agent_id)
                    if agent_bg:
                        reason = "embedding model" if embedding_changed else "drive folder"
                        print(f"[update_agent] {reason} changed for agent {stored_agent_id}. Re-indexing with force_reindex={embedding_changed}...")
                        # Force reindex if embedding model changed, otherwise just check for updates
                        reindex_agent(agent_bg, creds_path, track_progress=True, force_reindex=embedding_changed)
                        print(f"[update_agent] Background re-indexing completed for agent {stored_agent_id}")
                    else:
                        print(f"[update_agent] Agent {stored_agent_id} not found in background thread")
                        complete_progress(stored_agent_id, "Agent not found")
                except Exception as e:
                    error_msg = str(e)
                    print(f"[update_agent] Background re-indexing failed for agent {stored_agent_id}: {error_msg}")
                    import traceback
                    traceback.print_exc()
                    complete_progress(stored_agent_id, error_msg)
                finally:
                    db_bg.close()
            
            thread = threading.Thread(target=background_reindex, daemon=True)
            thread.start()
            
            # Redirect to progress page to show re-indexing progress
            return RedirectResponse(f"/agents/{a.id}/progress-page", status_code=302)
    
    # No embedding change, go back to chat
    return RedirectResponse(f"/a/{a.slug}", status_code=302)


@router.get("/{agent_id}/logs")
def view_logs(request: Request, agent_id: int, db: Session = Depends(get_db)):
    uid = get_current_user_id(request)
    agent = db.get(Agent, agent_id)

    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Security: Only the professor who owns this agent
    if agent.owner_id != uid:
        raise HTTPException(status_code=403, detail="Not allowed")

    logs = (
        db.query(QueryLog)
        .filter_by(agent_id=agent_id)
        .order_by(QueryLog.timestamp.desc())
        .all()
    )

    return templates.TemplateResponse(
        "agent_logs.html",
        {
            "request": request,
            "agent": agent,
            "logs": logs
        }
    )




@router.get("/{agent_id}/logs/download")
def download_logs(agent_id: int, request: Request, db: Session = Depends(get_db)):
    uid = get_current_user_id(request)
    agent = db.get(Agent, agent_id)

    if not agent or agent.owner_id != uid:
        raise HTTPException(status_code=404)

    logs = (
        db.query(QueryLog)
        .filter_by(agent_id=agent_id)
        .order_by(QueryLog.timestamp.asc())
        .all()
    )

    # Convert logs to list of dicts
    data = [
        {
            "timestamp": log.timestamp.isoformat(),
            "query": log.query,
        }
        for log in logs
    ]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as tmp:
        json.dump(data, tmp, indent=2)
        temp_path = tmp.name


    # Return file for download
    return FileResponse(
        tmp.name,
        media_type="application/json",
        filename=f"{agent.name}_logs.json"
    )


@router.get("/{agent_id}/progress")
def get_embedding_progress(agent_id: int, request: Request, db: Session = Depends(get_db)):
    """Get the current embedding progress for an agent"""
    uid = get_current_user_id(request)
    agent = db.get(Agent, agent_id)

    if not agent or agent.owner_id != uid:
        raise HTTPException(status_code=404)

    progress = get_progress(agent_id)
    if not progress:
        return JSONResponse({
            "status": "idle",
            "total_files": 0,
            "processed_files": 0,
            "current_file": "",
            "progress_percentage": 0
        })

    return JSONResponse({
        "status": progress.status,
        "total_files": progress.total_files,
        "processed_files": progress.processed_files,
        "current_file": progress.current_file,
        "progress_percentage": progress.progress_percentage,
        "error_message": progress.error_message
    })


@router.get("/{agent_id}/progress-page")
def show_progress_page(agent_id: int, request: Request, db: Session = Depends(get_db)):
    """Show the progress page for initial agent setup"""
    uid = get_current_user_id(request)
    agent = db.get(Agent, agent_id)

    if not agent or agent.owner_id != uid:
        raise HTTPException(status_code=404)

    return templates.TemplateResponse("agent_progress.html", {
        "request": request,
        "agent": agent
    })


@router.get("/{agent_id}/analytics")
def show_analytics_page(request: Request, agent_id: int, db: Session = Depends(get_db)):
    uid = get_current_user_id(request)
    agent = db.get(Agent, agent_id)

    if not agent or agent.owner_id != uid:
        raise HTTPException(status_code=403, detail="Not allowed")

    return templates.TemplateResponse("agent_logs.html", {
        "request": request,
        "agent": agent,
        "logs": []
    })


@router.get("/{agent_id}/analytics/data")
def analytics_data(agent_id: int, db: Session = Depends(get_db)):
    agent = db.get(Agent, agent_id)
    if not agent:
        raise HTTPException(status_code=404)

    queries = fetch_queries(db, agent_id)
    if not queries:
        return JSONResponse({"summary": None})

    provider = provider_from(agent)

    try:
        embeddings = provider.embed(queries)
    except Exception as e:
        print("[Analytics] Embedding error:", e)
        return JSONResponse({"summary": "Embedding failed."})

    clusters = cluster_queries(queries, embeddings, k=5)
    summary = summarize_clusters(clusters, provider)

    return JSONResponse({"summary": summary})