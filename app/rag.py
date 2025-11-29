# app/rag.py
from typing import List, Tuple, Dict
import os, json, shutil, logging
from pathlib import Path

from chromadb import Client
from chromadb.config import Settings as ChromaSettings
from chromadb.api.types import Documents, Embeddings  # typing hints
from typing import Optional
from pypdf import PdfReader

try:
    from langchain_google_community import GoogleDriveLoader  # preferred
except ImportError:
    from langchain_community.document_loaders import GoogleDriveLoader  # fallback

from .providers import OpenAIProvider, GeminiProvider, ProviderBase
from .settings import get_settings
import pdfplumber
import tempfile
import pytesseract
from PIL import Image
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import google.oauth2.credentials
import googleapiclient.discovery
import io
from googleapiclient.http import MediaIoBaseDownload
from .db import get_db
from .models import Agent, Conversation, Message, AgentFile
from .security import get_current_user_id
import json, time, threading
from .models import QueryLog
import uuid
from fastapi import APIRouter, Request, Depends, HTTPException, Form
from .db import SessionLocal
from .progress import start_progress, update_progress, complete_progress, is_force_reindex

log = logging.getLogger(__name__)

############################### ADDED ######################
db= SessionLocal()

def _ocr_image(img: Image.Image) -> str:
    try:
        return pytesseract.image_to_string(img)
    except Exception as e:
        print(f"[OCR] Tesseract error: {e}")
        return ""
    

def _list_all_images(folder_id: str, user_id: int):
    """
    List ONLY image files inside the given Google Drive folder.
    Uses a Drive API query that restricts results to image/* MIME types.
    """
    token_file = _user_token_path(user_id)

    creds = Credentials.from_authorized_user_file(
        str(token_file),
        ["https://www.googleapis.com/auth/drive.readonly"]
    )

    service = build("drive", "v3", credentials=creds)

    results = []
    page_token = None

    # Query ONLY images using mimeType contains 'image/'
    q = (
        f"'{folder_id}' in parents "
        f"and trashed = false "
        f"and mimeType contains 'image/'"
    )

    while True:
        response = service.files().list(
            q=q,
            fields="nextPageToken, files(id, name, mimeType, webContentLink, modifiedTime)",
            pageToken=page_token,
            includeItemsFromAllDrives=False,
            supportsAllDrives=True
        ).execute()

        results.extend(response.get("files", []))

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return results

def load_drive_images(agent) -> List[dict]:
    """
    Load ONLY image files from the folder using raw Drive API.
    PDF or other files are ignored by construction.
    """
    folder_id = file_id_from_url_or_id(agent.drive_folder_id or "")
    if not folder_id:
        return []

    # Raw API call that returns only images
    files = _list_all_images(folder_id, agent.owner_id)

    results = []
    seen_ids = set()

    for f in files:
        fid = f.get("id")
        if not fid or fid in seen_ids:
            continue
        seen_ids.add(fid)

        results.append({
            "id": fid,
            "source": f.get("webContentLink", "") or "",
            "title": f.get("name", ""),
            "mime": f.get("mimeType", "") or "",
            "is_image": True,       # always true
            "is_pdf": False,        # always false now
            "page": None,
            "slide": None,
            "last_modified": f.get("modifiedTime", "") or "",
            "text": "",             # text extracted later
        })

    # print("--- LOADED IMAGES FROM DRIVE ---")
    # for r in results:
    #     print(f"ID: {r['id']}")
    #     print(f"TITLE: {r['title']}")
    #     print(f"MIME: {r['mime']}")
    #     print("------------------------------")

    for img in results:
        if db.query(AgentFile).filter_by(agent_id=agent.id,  file_id=img["id"]).first():
            continue

        db.add(AgentFile(
            agent_id=agent.id,
            #file_id=str(uuid.uuid4()),
            file_id=img["id"],
            title=img["title"],
            page=None,
            #text=img["text"],
            last_modified=img["last_modified"]
        ))

    db.commit()

    return results



def download_drive_file(file_id: str, token_path: str) -> bytes:
    """
    Downloads a real file (PNG/JPG/PDF/etc) from Google Drive using OAuth token.
    Returns raw bytes.
    """
    # Load OAuth credentials
    creds_json = json.load(open(token_path, "r"))
    creds = google.oauth2.credentials.Credentials.from_authorized_user_info(
        creds_json,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )

    # Build Drive API client
    service = googleapiclient.discovery.build("drive", "v3", credentials=creds)

    # Request media
    request = service.files().get_media(fileId=file_id)

    # Download into memory
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()

    return buffer.getvalue()


########## 
def _extract_text_for_images(images: List[dict], owner_id: int) -> List[dict]:
    """
    Process a LIST of image documents and extract text via OCR.
    Each dict in the list is mutated in-place to include d["text"].
    """
    token_path = str(_user_token_path(owner_id))

    for d in images:
        mime = (d.get("mime") or "").lower()
        file_id = d.get("id")

        if not file_id or not mime.startswith("image/"):
            print(f"[WARN] Skipping non-image doc: {d.get('title')}")
            d["text"] = ""
            continue

        # If text already exists, skip
        if (d.get("text") or "").strip():
            d["text"] = _trim_for_embedding(d["text"])
            continue

        # Download file bytes
        try:
            file_bytes = download_drive_file(file_id, token_path)
        except Exception as e:
            print(f"[Drive] Failed to download image {file_id}: {e}")
            d["text"] = ""
            continue

        # OCR
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".img")
        try:
            tmp.write(file_bytes)
            tmp.close()

            try:
                img = Image.open(tmp.name)
                text = pytesseract.image_to_string(img)
                print(f"[IMAGE OCR] {d.get('title')} → {len(text)} chars")
            except Exception as e:
                print(f"[OCR] Failed on image {d.get('title')}: {e}")
                text = ""

        finally:
            try:
                os.remove(tmp.name)
            except:
                pass

        d["text"] = _trim_for_embedding(text)

    return images



# add this helper near the top (after imports)
def _sanitize_meta(meta: dict) -> dict:
    """Drop None and coerce everything else to a supported primitive."""
    out = {}
    for k, v in meta.items():
        if v is None:
            continue  # drop missing values entirely
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


# -------- Paths (absolute, auto-created) --------
S = get_settings()
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(S.DATA_DIR) if Path(S.DATA_DIR).is_absolute() else BASE_DIR / S.DATA_DIR
CHROMA_DIR = DATA_ROOT / "chroma"
TOKENS_DIR = DATA_ROOT / "google_tokens"
CLIENT_SECRETS_PATH = DATA_ROOT / "google_client_secret.json"  # <- place your OAuth client JSON here
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
TOKENS_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_embed_model(provider: str, raw: str) -> str:
    em = (raw or "").strip()
    if provider == "gemini":
        if em.startswith("openai:"):
            raise ValueError("Gemini provider cannot use OpenAI embedding model. Use 'models/text-embedding-004'.")
        if em.startswith("gemini:"):
            em = em.split(":", 1)[1]
        if em in ("", "text-embedding-3-small", "text-embedding-ada-002"):
            return "models/text-embedding-004"
        if not em.startswith("models/"):
            em = f"models/{em}"
        return em
    else:  # openai
        if em.startswith("gemini:") or em.startswith("models/"):
            raise ValueError("OpenAI provider cannot use Gemini embedding model. Use 'text-embedding-3-small'.")
        if em.startswith("openai:"):
            em = em.split(":", 1)[1]
        return em or "text-embedding-3-small"


def provider_from(agent) -> ProviderBase:
    api_key = agent.api_key if hasattr(agent, 'api_key') else None
    if agent.provider == "gemini":
        embed_model = _normalize_embed_model("gemini", agent.embed_model)
        return GeminiProvider(agent.model, embed_model, api_key)
    else:
        embed_model = _normalize_embed_model("openai", agent.embed_model)
        return OpenAIProvider(agent.model, embed_model, api_key)


class ProviderEmbeddingFunction:
    """Chroma 0.4.16+/0.5.x expects __call__(self, input) -> embeddings."""
    def __init__(self, provider: ProviderBase):
        self._p = provider

    def __call__(self, input: Documents) -> Embeddings:
        texts = list(input) if not isinstance(input, list) else input
        # Normalize to List[List[float]] so Chroma sees the correct dimensionality
        return _as_2d_floats(self._p.embed(texts))



def _embed_fn(provider: ProviderBase):
    return ProviderEmbeddingFunction(provider)


def _collection_name(agent_id: int) -> str:
    return f"agent_{agent_id}"


def get_vector_client():
    """Prefer Chroma 0.5+ PersistentClient; fall back to legacy Client(Settings)."""
    try:
        from chromadb import PersistentClient  # 0.5+
        return PersistentClient(
            path=str(CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
    except Exception:
        # 0.4.x fallback
        return Client(
            settings=ChromaSettings(
                anonymized_telemetry=False,
                persist_directory=str(CHROMA_DIR),
                allow_reset=True,
            )
        )


def _ensure_chroma_ready(client):
    """Create sys tables on fresh/migrated dirs; heal if 'tenants' table is missing."""
    try:
        client.heartbeat()
    except Exception as e:
        msg = str(e)
        if "no such table: tenants" in msg or "Could not connect to tenant" in msg:
            try:
                client.reset()
            except Exception:
                pass


def ensure_collection(agent, client=None):
    client = client or get_vector_client()
    name = _collection_name(agent.id)
    _ensure_chroma_ready(client)
    
    provider = provider_from(agent)
    embed_fn = _embed_fn(provider)
    
    # Check if collection exists
    existing_collections = [c.name for c in client.list_collections()]
    
    if name in existing_collections:
        # Collection exists - try to get it and handle dimension mismatch
        try:
            col = client.get_collection(name=name, embedding_function=embed_fn)
            return col
        except Exception as e:
            # Dimension mismatch or other error - delete and recreate
            error_msg = str(e).lower()
            if "dimension" in error_msg or "dimensionality" in error_msg:
                print(f"[ensure_collection] Dimension mismatch for agent {agent.id}. Deleting old collection...")
            else:
                print(f"[ensure_collection] Error getting collection for agent {agent.id}: {e}")
            
            try:
                client.delete_collection(name)
                print(f"[ensure_collection] Deleted collection '{name}'. Creating new one...")
            except Exception as del_err:
                print(f"[ensure_collection] Failed to delete collection: {del_err}")
    
    # Create new collection
    try:
        col = client.create_collection(
            name=name,
            embedding_function=embed_fn,
        )
        print(f"[ensure_collection] Created collection '{name}' for agent {agent.id}")
        return col
    except Exception as create_error:
        print(f"[ensure_collection] Failed to create collection: {create_error}")
        raise


# ---------------- Google Drive auth helpers ----------------
def _user_token_path(user_id: int) -> Path:
    """We store per-user OAuth or service account JSON at data/google_tokens/user_<id>.json"""
    return TOKENS_DIR / f"user_{user_id}.json"


def _build_drive_auth_kwargs(user_id: int) -> Dict:
    """
    Returns kwargs for GoogleDriveLoader:
      - Service account: {'credentials_path': '<sa.json>'}
      - OAuth installed app: {'token_path': '<user_token.json>', 'client_secrets_path': '<client_secret.json>', 'scopes': [..]}
    If neither is available, returns {} (loader will fallback to ADC and likely fail).
    """
    kw: Dict = {}
    token_file = _user_token_path(user_id)
    if token_file.exists():
        try:
            blob = json.loads(token_file.read_text(encoding="utf-8"))
        except Exception:
            blob = {}

        # Service account JSON has "type": "service_account"
        if blob.get("type") == "service_account":
            kw["credentials_path"] = str(token_file)
            # TIP: Make sure the Drive folder is shared with the service account email.
            log.info("Using service account credentials at %s", token_file)
            return kw

        # Otherwise assume it's an OAuth "authorized_user" token JSON (refresh_token + client info NOT included)
        # The loader needs both the token AND the client secrets.
        if CLIENT_SECRETS_PATH.exists():
            kw["token_path"] = str(token_file)              # your existing token json (authorized_user)
            kw["client_secrets_path"] = str(CLIENT_SECRETS_PATH)  # the OAuth client secrets json
            kw["scopes"] = ["https://www.googleapis.com/auth/drive.readonly"]
            log.info("Using OAuth token at %s with client secrets %s", token_file, CLIENT_SECRETS_PATH)
            return kw
        else:
            log.warning(
                "Found OAuth token at %s but no client secrets at %s. "
                "Place your OAuth client secret JSON there (or switch to a service account).",
                token_file, CLIENT_SECRETS_PATH
            )

    # Nothing found; will fallback to ADC inside loader (likely to error if ADC not set)
    log.warning("No Google credentials found for user %s. Falling back to ADC.", user_id)
    return kw


# ---------------- Google Drive loading ----------------
def file_id_from_url_or_id(val: str) -> str:
    if not val:
        return ""
    if "folders/" in val:
        return val.split("folders/")[-1].split("?")[0].strip("/ ")
    if "open?id=" in val:
        return val.split("open?id=")[-1].split("&")[0]
    return val  # already an ID


def validate_drive_folder(drive_folder_url: str, user_id: int) -> Tuple[bool, str]:
    """
    Validate that a Google Drive folder exists, is accessible, and contains files.
    
    Args:
        drive_folder_url: The Google Drive folder URL or ID
        user_id: The user's ID (to get their OAuth credentials)
        
    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    # Extract folder ID from URL
    folder_id = file_id_from_url_or_id(drive_folder_url)
    if not folder_id:
        return False, "Invalid Google Drive folder URL. Please provide a valid folder link."
    
    # Get user credentials
    token_file = _user_token_path(user_id)
    if not token_file.exists():
        return False, "Google Drive not connected. Please connect your Google Drive first."
    
    try:
        creds = Credentials.from_authorized_user_file(
            str(token_file),
            ["https://www.googleapis.com/auth/drive.readonly"]
        )
        service = build("drive", "v3", credentials=creds)
        
        # First, check if the folder itself exists and is accessible
        try:
            folder_meta = service.files().get(
                fileId=folder_id,
                fields="id, name, mimeType"
            ).execute()
            
            # Verify it's actually a folder
            if folder_meta.get("mimeType") != "application/vnd.google-apps.folder":
                return False, "The provided link is not a folder. Please provide a Google Drive folder link."
                
        except Exception as e:
            error_str = str(e).lower()
            if "404" in error_str or "not found" in error_str:
                return False, "Google Drive folder not found. Please check the folder link and ensure it's shared with your Google account."
            elif "403" in error_str or "forbidden" in error_str or "permission" in error_str:
                return False, "Access denied to the Google Drive folder. Please ensure the folder is shared with your Google account."
            else:
                log.error(f"[validate_drive_folder] Error checking folder: {e}")
                return False, f"Could not access Google Drive folder: {str(e)}"
        
        # Check if the folder contains any files
        response = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="files(id, name, mimeType)",
            pageSize=10,  # We only need to know if there's at least one file
            includeItemsFromAllDrives=False,
            supportsAllDrives=True
        ).execute()
        
        files = response.get("files", [])
        if not files:
            return False, "The Google Drive folder is empty. Please add course materials to the folder before creating an agent."
        
        # Success! Folder exists, is accessible, and has files
        log.info(f"[validate_drive_folder] Folder '{folder_meta.get('name')}' validated: {len(files)}+ files found")
        return True, ""
        
    except Exception as e:
        log.error(f"[validate_drive_folder] Unexpected error: {e}")
        return False, f"Error validating Google Drive folder: {str(e)}"


def _load_drive_docs_raw(folder_id: str, loader_auth_kwargs: Dict):
    """Try to load everything; if pypdf is missing, retry without PDFs."""
    loader = GoogleDriveLoader(
        folder_id=folder_id,
        recursive=True,
        file_loader_cls=None,
        **loader_auth_kwargs,
    )
    try:
        return loader.load()
    except ModuleNotFoundError as e:
        msg = str(e)
        if "PyPDF2" in msg or "pypdf" in msg:
            log.warning("pypdf/PyPDF2 not installed; skipping PDFs. Install 'pypdf' for PDF parsing.")
            try:
                loader = GoogleDriveLoader(
                    folder_id=folder_id,
                    recursive=True,
                    file_loader_cls=None,
                    file_types=["document", "sheet", "pdf", "presentation"],
                    **loader_auth_kwargs,
                )
                return loader.load()
            except TypeError:
                log.warning("GoogleDriveLoader lacks 'file_types'; returning no docs. Install 'pypdf' or newer 'langchain-google-community'.")
                return []
        raise


def retrieve(agent, query: str, k: int = 30) -> List[dict]:
    client = get_vector_client()
    col = ensure_collection(agent, client)
    
    try:
        # Get collection count to avoid requesting more results than available
        col_count = col.count()
        if col_count == 0:
            return []
        
        # Limit k to the number of documents in the collection
        actual_k = min(k, col_count)
        
        res = col.query(query_texts=[query], n_results=max(1, actual_k))
    except Exception as e:
        error_msg = str(e).lower()
        
        # Handle HNSW index errors (ef or M too small, contiguous array errors)
        if "contiguous" in error_msg or "ef or m" in error_msg:
            print(f"[retrieve] HNSW index error. Trying with fewer results...")
            try:
                # Try with minimal results
                res = col.query(query_texts=[query], n_results=1)
            except:
                print(f"[retrieve] Query still failed. Returning no results.")
                return []
        # Handle dimension mismatch during query
        elif "dimension" in error_msg or "dimensionality" in error_msg:
            print(f"[retrieve] Dimension mismatch during query. Recreating collection for agent {agent.id}...")
            # Delete and recreate collection
            name = _collection_name(agent.id)
            try:
                client.delete_collection(name)
            except:
                pass
            # Recreate and retry
            col = ensure_collection(agent, client)
            # If collection is empty, return empty results
            try:
                res = col.query(query_texts=[query], n_results=max(1, k))
            except:
                print(f"[retrieve] Collection is empty after recreation. Returning no results.")
                return []
        else:
            raise

    if not res or not res.get("ids") or not res["ids"][0]:
        return []
    

    out = []
    ids = res["ids"][0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    for i in range(len(ids)):
        out.append({
            "id": ids[i],
            "text": docs[i] if i < len(docs) else "",
            "metadata": metas[i] if i < len(metas) else {},
        })
    return out


# ---------------- Google Drive loading ----------------
def _make_doc_id(agent_id: int, md: dict, idx: int) -> str:
    """
    Make a stable, unique ID per returned Document.
    Priority:
      1) metadata['source'] (Drive file URL/id)
      2) Document.id
      3) fallback to agent_id
    Then append page/slide index if present; otherwise append running index.
    """
    base = md.get("source") or md.get("id") or f"agent-{agent_id}"
    # Some loaders provide these keys for pagination:
    page = md.get("page")
    slide = md.get("slide")
    suffix_parts = []
    if page is not None:
        suffix_parts.append(f"p{page}")
    if slide is not None:
        suffix_parts.append(f"s{slide}")
    if suffix_parts:
        return f"{base}#{'-'.join(suffix_parts)}"
    # Final fallback so each entry is unique even if metadata lacks page/slide
    return f"{base}#i{idx}"


# def load_drive_docs(agent) -> List[dict]:
#     """Load Google Drive docs (recursive) and normalize."""
#     folder_id = file_id_from_url_or_id(agent.drive_folder_id or "")
#     auth_kwargs = _build_drive_auth_kwargs(agent.owner_id)
#     docs_raw = _load_drive_docs_raw(folder_id, auth_kwargs)

#     results = []
#     seen_ids = set()
#     for i, d in enumerate(docs_raw):
#         md = getattr(d, "metadata", {}) or {}
#         text = getattr(d, "page_content", "") or ""

#         doc_id = _make_doc_id(agent.id, md, i)
#         print("[DEBUG] doc_id ", doc_id)
#         # Guard against any accidental dup even after suffixing
#         if doc_id in seen_ids:
#             doc_id = f"{doc_id}#dup{i}"
#         seen_ids.add(doc_id)

#         results.append({
#             "id": doc_id,
#             "source": md.get("source", ""),
#             "title": md.get("title", ""),
#             "page": md.get("page"),
#             "slide": md.get("slide"),
#             "last_modified": md.get("last_modified_time", "") or md.get("modified_time", ""),
#             "text": text,
#         })

#         for doc in results:
#                 if db.query(AgentFile).filter_by(agent_id=agent.id,  file_id=doc["id"]).first():
#                     continue

#                 db.add(AgentFile(
#                     agent_id=agent.id,
#                     #file_id=str(uuid.uuid4()),
#                     file_id=doc["id"],
#                     title=doc["title"],
#                     page=None,
#                     #text=img["text"],
#                     last_modified=doc["last_modified"]
#                 ))

#         db.commit()

#     return results


def _list_all_files_in_drive_folder(folder_id: str, user_id: int):
    """
    List ALL files inside the given Google Drive folder using raw Drive API.
    Returns list of file metadata dicts.
    """
    token_file = _user_token_path(user_id)

    creds = Credentials.from_authorized_user_file(
        str(token_file),
        ["https://www.googleapis.com/auth/drive.readonly"]
    )

    service = build("drive", "v3", credentials=creds)

    results = []
    page_token = None

    while True:
        response = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, mimeType, webContentLink, modifiedTime)",
            pageToken=page_token,
            includeItemsFromAllDrives=False,
            supportsAllDrives=True
        ).execute()

        results.extend(response.get("files", []))

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return results


def _extract_text_from_pdf(file_bytes: bytes, title: str = "") -> str:
    """
    Extract text from PDF bytes with a robust two-stage strategy:

      1) Fast path: pypdf.PdfReader over the whole file.
         - If it returns any non-empty text, use that and STOP (no OCR).
      2) Fallback: pdfplumber per-page.
         - For each page, try page.extract_text().
         - If no text or extract_text() fails, fallback to OCR using
           page.to_image() + Tesseract.

    This gives us:
      - fast selectable-text extraction when possible (like on Windows),
      - OCR only when necessary (scanned/image-only PDFs or broken text extractors).
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        tmp.write(file_bytes)
        tmp.close()

        # -------------------------------
        # 1) Fast path: pypdf.PdfReader
        # -------------------------------
        try:
            reader = PdfReader(tmp.name)
            fast_text = ""
            for i, page in enumerate(reader.pages):
                try:
                    t = page.extract_text() or ""
                    fast_text += t + "\n"
                except Exception as e2:
                    print(f"[PDF pypdf] Failed on page {i+1} of '{title}': {e2}")

            if fast_text.strip():
                # We got real text, no need to run pdfplumber or OCR.
                print(
                    f"[PDF] Extracted total {len(fast_text)} chars from '{title}' "
                    f"(pypdf fast path)"
                )
                return fast_text
            else:
                print(
                    f"[PDF] pypdf fast path extracted no text from '{title}', "
                    "falling back to pdfplumber + OCR"
                )
        except Exception as e:
            print(
                f"[PDF] pypdf fast path failed for '{title}': {e}. "
                "Falling back to pdfplumber + OCR."
            )

        # ---------------------------------------
        # 2) Fallback: pdfplumber + page-level OCR
        # ---------------------------------------
        text = ""
        try:
            with pdfplumber.open(tmp.name) as pdf:
                for i, page in enumerate(pdf.pages):
                    # Try selectable text first
                    extracted = ""
                    try:
                        extracted = page.extract_text() or ""
                    except Exception as e:
                        print(
                            f"[PDF] extract_text failed on page {i+1} of "
                            f"'{title}': {e}"
                        )
                        extracted = ""

                    if extracted.strip():
                        # Normal selectable text
                        text += extracted + "\n"
                    else:
                        # No selectable text -> OCR fallback using page.to_image()
                        try:
                            pil_page = page.to_image(resolution=300).original
                            ocr_text = pytesseract.image_to_string(pil_page)
                            if ocr_text and ocr_text.strip():
                                text += ocr_text + "\n"
                                print(
                                    f"[PDF OCR] Page {i+1} of '{title}': "
                                    f"extracted {len(ocr_text)} chars via OCR (fallback)"
                                )
                            else:
                                print(
                                    f"[PDF OCR] Page {i+1} of '{title}': "
                                    "no text found even after OCR"
                                )
                        except Exception as e:
                            print(
                                f"[PDF OCR] Failed on page {i+1} of '{title}': {e}"
                            )

            print(f"[PDF] Extracted total {len(text)} chars from '{title}' (pdfplumber+OCR)")
            return text

        except Exception as e:
            # If pdfplumber as a whole explodes, we have no further fallback here.
            print(f"[PDF] Failed to process PDF '{title}' with pdfplumber: {e}")
            return ""

    finally:
        try:
            os.remove(tmp.name)
        except:
            pass



def _extract_text_from_image(file_bytes: bytes, title: str = "") -> str:
    """Extract text from image bytes using OCR."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".img")
    try:
        tmp.write(file_bytes)
        tmp.close()

        try:
            img = Image.open(tmp.name)
            text = pytesseract.image_to_string(img)
            print(f"[Image OCR] Extracted {len(text)} chars from '{title}'")
            return text
        except Exception as e:
            print(f"[Image OCR] Failed on '{title}': {e}")
            return ""

    finally:
        try:
            os.remove(tmp.name)
        except:
            pass


def load_drive_docs(agent, track_progress: bool = False, force_reindex: bool = False) -> List[dict]:
    """
    Load Google Drive docs using raw Drive API and extract text with OCR support.
    Handles PDFs (selectable and non-selectable), images, and other formats.
    
    Args:
        agent: The agent to load documents for
        track_progress: Whether to track progress for UI display
        force_reindex: If True, indicates this is a force reindex (for progress tracking purposes)
    """
    folder_id = file_id_from_url_or_id(agent.drive_folder_id or "")
    if not folder_id:
        return []

    # List all files using raw Drive API
    files = _list_all_files_in_drive_folder(folder_id, agent.owner_id)
    
    # Initialize progress tracking if requested
    if track_progress:
        start_progress(agent.id, len(files), force_reindex=force_reindex)
    
    token_path = str(_user_token_path(agent.owner_id))
    
    results = []
    seen_ids = set()
    processed_count = 0

    for f in files:
        file_id = f.get("id")
        if not file_id or file_id in seen_ids:
            continue
        seen_ids.add(file_id)

        name = f.get("name", "")
        mime = f.get("mimeType", "") or ""
        web_link = f.get("webContentLink", "") or ""
        modified = f.get("modifiedTime", "") or ""

        is_image = mime.startswith("image/")
        is_pdf = (mime == "application/pdf")

        # Initialize text as empty
        text = ""
        
        # Download and extract text for PDFs and images
        if is_pdf or is_image:
            try:
                file_bytes = download_drive_file(file_id, token_path)
                
                if is_pdf:
                    text = _extract_text_from_pdf(file_bytes, name)
                elif is_image:
                    text = _extract_text_from_image(file_bytes, name)
                    
            except Exception as e:
                print(f"[Drive] Failed to download/process '{name}': {e}")
                text = ""

        results.append({
            "id": file_id,
            "source": web_link,
            "title": name,
            "page": None,
            "slide": None,
            "mime": mime,
            "is_image": is_image,
            "is_pdf": is_pdf,
            "last_modified": modified,
            "text": text,
        })
        
        # Update progress after processing each file
        if track_progress:
            processed_count += 1
            update_progress(agent.id, processed_count, name)

    return results



# -------- Indexing & Retrieval --------
def _trim_for_embedding(text: str, limit: int = 12000) -> str:
    # keep well under typical embedding limits; adjust if you know your model's exact cap
    t = (text or "").strip()
    return t[:limit] if len(t) > limit else t


def reindex_agent(agent, _creds_path_ignored: str = "", track_progress: bool = True, force_reindex: bool = False) -> Tuple[int, int, int]:
    """
    Reindex an agent's documents.
    
    Args:
        agent: The agent to reindex
        _creds_path_ignored: Deprecated parameter, kept for compatibility
        track_progress: Whether to track progress for UI display
        force_reindex: If True, forces full re-embedding regardless of cache
    
    Note: API key validation is done upfront before agent creation/update,
          so we assume the API key is valid when this function is called.
    """
    try:
        # Load all documents (PDFs, images, etc.) with OCR support
        # Progress tracking starts inside load_drive_docs
        all_docs = load_drive_docs(agent, track_progress=track_progress, force_reindex=force_reindex)
        
        if not all_docs:
            # No documents found - still mark as complete
            print(f"[reindex_agent] No documents found in Drive folder for agent {agent.id}")
            complete_progress(agent.id)
            return (0, 0, 0)
        
        client = get_vector_client()
        
        # Check if force reindex is requested (e.g., embedding model changed)
        # If force_reindex, delete existing collection first
        if force_reindex:
            print(f"[reindex_agent] Force reindex requested for agent {agent.id}. Deleting existing collection...")
            name = _collection_name(agent.id)
            try:
                client.delete_collection(name)
                print(f"[reindex_agent] Deleted collection '{name}'")
            except Exception as del_err:
                print(f"[reindex_agent] Note: Could not delete collection (may not exist): {del_err}")
        
        col = ensure_collection(agent, client)
        provider = provider_from(agent)

        try:
            existing_ids = set(col.get()['ids'])
            existing_items = col.get()
            existing_map = {}
            for _id, meta in zip(existing_items["ids"], existing_items["metadatas"]):
                existing_map[_id] = meta
        except Exception as e:
            # Handle dimension mismatch or empty collection
            if "dimension" in str(e).lower() or "dimensionality" in str(e).lower():
                print(f"[reindex_agent] Dimension mismatch detected. Recreating collection for agent {agent.id}...")
                name = _collection_name(agent.id)
                try:
                    client.delete_collection(name)
                except:
                    pass
                col = ensure_collection(agent, client)
            # Assume empty collection
            existing_ids = set()
            existing_map = {}

        current_ids = set([d["id"] for d in all_docs])
        to_delete = existing_ids - current_ids

        # ----------------------------
        # EXPANDED DELETE FIX  
        # ----------------------------
        # Ensure the base file source is marked for deletion
        # Any file removed from Drive must have its source removed too
        removed_sources = set()
        for eid in to_delete:
            meta = existing_map.get(eid)
            if meta and meta.get("source"):
                removed_sources.add(meta["source"])

        for eid, meta in existing_map.items():
            if meta.get("source") in removed_sources:
                to_delete.add(eid)

        expanded_delete = set(to_delete)
        base_ids = {eid.split("#")[0] for eid in to_delete}

        for eid in existing_ids:
            base = eid.split("#")[0]
            if base in base_ids:
                expanded_delete.add(eid)

        if expanded_delete:
            print("Deleting expanded set:", expanded_delete)
            col.delete(ids=list(expanded_delete))

            # ----------------------------
            # DELETE FROM LOCAL DATABASE  
            # ----------------------------
            session = SessionLocal()

            for eid in expanded_delete:
                db_entry = session.query(AgentFile).filter_by(agent_id=agent.id, file_id=eid).first()
                if db_entry:
                    print(f"[DB] Deleting AgentFile entry: {eid}")
                    session.delete(db_entry)

            session.commit()
            session.close()

        else:
            print("No items to delete from Chroma (expanded delete)")

        # Determine which documents need to be processed
        dirty_docs = []
        
        # If force reindex, ALL documents are dirty
        if force_reindex:
            dirty_docs = all_docs
            print(f"[reindex_agent] Force reindex: processing all {len(dirty_docs)} documents")
        else:
            for d in all_docs:
                fid = d["id"]
                modified = d.get("last_modified", "")

                if fid not in existing_ids:
                    dirty_docs.append(d)
                    continue

                stored_meta = existing_map.get(fid)

                if stored_meta and stored_meta.get("last_modified") != modified:
                    dirty_docs.append(d)
                    continue
        if not dirty_docs:
            print("[reindex_agent] No new or updated files. Nothing to index.")
            complete_progress(agent.id)
            return (0, 0, 0)


        # 1) Build triples
        triples = []
        dropped_empty = 0
        
        for d in dirty_docs:
            text = _trim_for_embedding(d.get("text") or "")
            title = d.get("title") or ""
            is_img = d.get("is_image", False)

            # prepend metadata into searchable text
            meta_prefix = f"FILE: {title}. IMAGE: {is_img}. "
            text = meta_prefix + _trim_for_embedding(d.get("text") or "")

            if not text:
                dropped_empty += 1
                continue
            meta = _sanitize_meta({
                "title": d.get("title") or "",
                "source": d.get("source") or "",
                "page": d.get("page"),
                "slide": d.get("slide"),
                "last_modified": d.get("last_modified") or "",
                "is_image": d.get("is_image", False),
                "is_pdf": d.get("is_pdf", False),
            })
            triples.append((d["id"], meta, text))
            print ("[DEBUG] META:",meta)

        if dropped_empty:
            print(f"[reindex_agent] Skipped {dropped_empty} empty chunks out of {len(all_docs)}")
        if not triples:
            complete_progress(agent.id)
            return (0, 0, 0)

        # 2) Embed & upsert in batches
        BATCH = 64
        added = 0
        salvaged_total = 0
        dims = None

        for i in range(0, len(triples), BATCH):
            batch = triples[i:i+BATCH]
            ids_batch   = [t[0] for t in batch]
            metas_batch = [t[1] for t in batch]
            docs_batch  = [t[2] for t in batch]

            # Try batch embed
            embs_batch = None
            try:
                embs_batch = _as_2d_floats(provider.embed(docs_batch))
            except Exception as e:
                print(f"[reindex_agent] Batch embed error at [{i}:{i+len(batch)}]: {e}")

            # Salvage per item if needed
            need_salvage = (
                embs_batch is None or
                not isinstance(embs_batch, (list, tuple)) or
                len(embs_batch) != len(docs_batch)
            )
            if not need_salvage:
                d0 = len(embs_batch[0])
                if any(len(v) != d0 for v in embs_batch):
                    need_salvage = True

            if need_salvage:
                keep_idx, fixed_embs = [], []
                for j, txt in enumerate(docs_batch):
                    try:
                        one = provider.embed([txt])
                        one = _as_2d_floats(one)[0]
                        if dims is None:
                            dims = len(one)
                        if len(one) != dims:
                            print(f"[reindex_agent] Dropped item {i+j}: dim {len(one)} != {dims}")
                            continue
                        fixed_embs.append(one)
                        keep_idx.append(j)
                    except Exception as ee:
                        print(f"[reindex_agent] Dropped item {i+j}: {ee}")
                if not keep_idx:
                    continue
                ids_batch   = [ids_batch[j] for j in keep_idx]
                metas_batch = [metas_batch[j] for j in keep_idx]
                docs_batch  = [docs_batch[j] for j in keep_idx]
                embs_batch  = fixed_embs
                salvaged_total += (len(batch) - len(keep_idx))
            else:
                dims = dims or len(embs_batch[0])

            col.upsert(ids=ids_batch, documents=docs_batch, metadatas=metas_batch, embeddings=embs_batch)
            added += len(ids_batch)

        if salvaged_total:
            print(f"[reindex_agent] Salvaged {salvaged_total} items via per-item embedding fallback")

        complete_progress(agent.id)
        return (added, 0, added)
        
    except Exception as e:
        # Mark progress as failed
        complete_progress(agent.id, str(e))
        raise

def _flatten_once(x):
    # If x == [[...]] (extra wrapper), unwrap one level.
    if isinstance(x, (list, tuple)) and len(x) == 1 and isinstance(x[0], (list, tuple)):
        return x[0]
    return x

def _as_2d_floats(embs):
    """
    Normalize provider output to List[List[float]].
    Handles numpy arrays, dicts with 'embedding', extra nesting, etc.
    """
    # Provider may return dict: {"data":[{"embedding":[...]} , ...]}
    if isinstance(embs, dict) and "data" in embs:
        embs = [d.get("embedding") for d in embs["data"]]

    if not isinstance(embs, (list, tuple)):
        raise TypeError(f"Embeddings must be a list, got {type(embs)}")

    out = []
    for e in embs:
        # Some SDKs return {"embedding":[...]} per item
        if isinstance(e, dict) and "embedding" in e:
            e = e["embedding"]
        # numpy -> list
        if hasattr(e, "tolist"):
            e = e.tolist()
        # If extra wrapper (e == [[floats]]), unwrap once
        e = _flatten_once(e)
        # If somehow it's still nested (ragged), flatten one level
        if len(e) > 0 and isinstance(e[0], (list, tuple)):
            e = [float(v) for sub in e for v in (sub.tolist() if hasattr(sub, "tolist") else sub)]
        else:
            e = [float(v) for v in e]
        out.append(e)
    return out

