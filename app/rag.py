from typing import List, Tuple, Dict
import os, json, shutil, logging
from pathlib import Path
import numpy as np

from chromadb import Client
from chromadb.config import Settings as ChromaSettings
from chromadb.api.types import Documents, Embeddings  # typing hints
from typing import Optional

try:
    from langchain_google_community import GoogleDriveLoader  # preferred
except ImportError:
    from langchain_community.document_loaders import GoogleDriveLoader  # fallback

from .providers import OpenAIProvider, GeminiProvider, ProviderBase

from .settings import get_settings

log = logging.getLogger(__name__)

# def _as_2d_floats(embs):
#     """
#     Convert embeddings to 2D list of floats.
#     Handles single vectors, nested vectors, or irregular structures.
#     """
#     def flatten_vec(vec):
#         """Flatten nested lists/tuples into a flat list of floats."""
#         out = []
#         for x in vec:
#             if isinstance(x, (list, tuple)):
#                 out.extend(flatten_vec(x))
#             else:
#                 out.append(float(x))
#         return out

#     if not embs:
#         return []

#     # If top-level is a single vector (numbers), wrap in list
#     if all(isinstance(x, (int, float)) for x in embs):
#         return [list(map(float, embs))]

#     # Otherwise, flatten each embedding
#     return [flatten_vec(v) for v in embs]
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
##################################OCR
import pdfplumber
import requests
import tempfile
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import mimetypes
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

def _ocr_image(img: Image.Image) -> str:
    try:
        return pytesseract.image_to_string(img)
    except Exception as e:
        print(f"[OCR] Tesseract error: {e}")
        return ""


import google.oauth2.credentials
import googleapiclient.discovery
import io
from googleapiclient.http import MediaIoBaseDownload


def _pdf_has_text_layer(path):
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    return True
    except:
        pass
    return False

def _pdf_to_images(path):
    return convert_from_path(path, dpi=200)


def _extract_text_scanned_pdf(path):
    pages = _pdf_to_images(path)
    text = ""
    for img in pages:
        text += _ocr_image(img) + "\n"
    return text

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
   provider = (agent.provider or "").lower()
   if provider == "gemini":
       embed_model = _normalize_embed_model("gemini", agent.embed_model)
       return GeminiProvider(agent.model, embed_model)
   elif provider == "openai":
       embed_model = _normalize_embed_model("openai", agent.embed_model)
       return OpenAIProvider(agent.model, embed_model)
   else:
       raise ValueError(f"Unknown provider: {agent.provider}")



class ProviderEmbeddingFunction:
    """Chroma 0.4.16+/0.5.x expects __call__(self, input) -> embeddings."""
    def __init__(self, provider: ProviderBase):
        self._p = provider

    def __call__(self, input: Documents) -> Embeddings:
            texts = list(input) if not isinstance(input, list) else input
            emb = self._p.embed(texts)

            # ---- Normalize all shapes to 2D float list ----

            # Case 1: numpy array -> convert to list
            if isinstance(emb, np.ndarray):
                emb = emb.tolist()

            # Case 2: single vector: [0.1, 0.2, ...]
            if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], (float, int)):
                return [[float(x) for x in emb]]

            # Case 3: triple nested: [[[0.1, 0.2, ...]]]
            if (
                isinstance(emb, list)
                and len(emb) == 1
                and isinstance(emb[0], list)
                and len(emb[0]) == 1
                and isinstance(emb[0][0], list)
            ):
                emb = emb[0]   # remove first nesting

            # Case 4: normal 2D list → convert values to floats
            return [[float(x) for x in vec] for vec in emb]

def _list_all_files_r(folder_id: str, user_id: int):
    """
    List ONLY files inside the given Google Drive folder.
    Handles pagination and filters out trashed items.
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

def load_drive_docs_r(agent) -> List[dict]:
    """
    List *all* files in the Drive folder via raw Drive API
    and return lightweight doc records.

    Actual text extraction (PDF parsing, OCR, etc.) will be done later
    in reindex_agent, based on mime type.
    """
    folder_id = file_id_from_url_or_id(agent.drive_folder_id or "")
    if not folder_id:
        return []

    # Use our raw Drive API helper (already defined at top)
    files = _list_all_files_r(folder_id, agent.owner_id)

    results = []
    seen_ids = set()

    for f in files:
        fid = f.get("id")
        if not fid or fid in seen_ids:
            continue
        seen_ids.add(fid)

        name = f.get("name", "")
        mime = f.get("mimeType", "") or ""
        web_link = f.get("webContentLink", "") or ""
        modified = f.get("modifiedTime", "") or ""

        is_image = mime.startswith("image/")
        is_pdf = (mime == "application/pdf")

        results.append({
            "id": fid,                  # pure Drive file ID
            "source": web_link,         # for traceability / logging
            "title": name,
            "page": None,
            "slide": None,
            "mime": mime,
            "is_image": is_image,
            "is_pdf": is_pdf,
            "last_modified": modified,
            "text": "",                 # we'll fill in reindex_agent
        })

    print("--- LOADED DOCS FROM DRIVE (RAW API) ---")
    for r in results:
        print(f"ID: {r['id']}")
        print(f"TITLE: {r['title']}")
        print(f"MIME: {r['mime']}")
        print("------------------------------")

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


def _extract_text_for_doc(d: dict, owner_id: int) -> str:
    """
    Given a doc record from load_drive_docs, download the file if needed
    and extract text based on its mime type.

    Supported:
      - PDFs (native or scanned)
      - Images (PNG/JPG/JPEG/...)
      - Plain text
    """
    # If loader already gave us text (for future hybrid usage), keep it.
    existing = d.get("text") or ""
    if existing.strip():
        return _trim_for_embedding(existing)

    mime = (d.get("mime") or "").lower()
    file_id = d.get("id")
    if not file_id:
        return ""

    # Path to OAuth token for this user
    token_path = str(_user_token_path(owner_id))

    try:
        # Download raw bytes using your helper
        file_bytes = download_drive_file(file_id, token_path)
    except Exception as e:
        print(f"[Drive] Failed to download file {file_id}: {e}")
        return ""

    # --------- PDF ---------
    if mime == "application/pdf":
        # Write bytes to a temp .pdf file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        try:
            tmp.write(file_bytes)
            tmp.close()

            if _pdf_has_text_layer(tmp.name):
                # Use pdfplumber to get text
                try:
                    text = ""
                    with pdfplumber.open(tmp.name) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text() or ""
                            text += page_text + "\n"
                except Exception as e:
                    print(f"[PDF] Error extracting text from {d.get('title')}: {e}")
                    text = ""
            else:
                # Scanned PDF → OCR each page
                text = _extract_text_scanned_pdf(tmp.name)

        finally:
            try:
                os.remove(tmp.name)
            except:
                pass
        d["text"] = text
        return _trim_for_embedding(text)

    # --------- Images (PNG/JPG/...) ---------
    if mime.startswith("image/"):
        print("Extracting text from image ")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".img")
        try:
            tmp.write(file_bytes)
            tmp.close()

            try:
                img = Image.open(tmp.name)
                text = pytesseract.image_to_string(img)
                print("text from image : " , text)
            except Exception as e:
                print(f"[OCR] Error processing image {d.get('title')}: {e}")
                text = ""

        finally:
            try:
                os.remove(tmp.name)
            except:
                pass
        d["text"] = text
        return _trim_for_embedding(text)

    # --------- Plain text ---------
    if mime == "text/plain":
        try:
            text = file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            text = ""
        d["text"] = text
        return _trim_for_embedding(text)

    # --------- Other types (optional TODO: docs/slides export) ---------
    # For now, ignore unsupported mimetypes or handle later (e.g. export Google Docs as PDF)
    print(f"[WARN] Unsupported mime type for text extraction: {mime} (file: {d.get('title')})")
    return ""

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
    try:
        col = client.get_or_create_collection(
            name=name,
            embedding_function=_embed_fn(provider_from(agent)),
        )
    except Exception as e:
        if "no such table: tenants" in str(e) or "Could not connect to tenant" in str(e):
            _ensure_chroma_ready(client)
        existing = [c.name for c in client.list_collections()]
        if name in existing:
            client.delete_collection(name)
        col = client.get_or_create_collection(
            name=name,
            embedding_function=_embed_fn(provider_from(agent)),
        )
    return col




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


def retrieve(agent, query: str, k: int = 5) -> List[dict]:
    client = get_vector_client()
    col = ensure_collection(agent, client)
    res = col.query(query_texts=[query], n_results=max(1, k))

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


def load_drive_docs(agent) -> List[dict]:
    """
    Load Google Drive docs (recursive) and return ONLY new/updated docs.
    Uses agent.index_state to remember last indexed timestamps.
    """

    # 1. Previous state (store last modified timestamps)
    indexed = getattr(agent, "index_state", {}) or {}

    folder_id = file_id_from_url_or_id(agent.drive_folder_id or "")
    auth_kwargs = _build_drive_auth_kwargs(agent.owner_id)

    # 2. Load everything from Drive
    docs_raw = _load_drive_docs_raw(folder_id, auth_kwargs)

    results = []
    seen_ids = set()

    for i, d in enumerate(docs_raw):
        md = getattr(d, "metadata", {}) or {}
        text = getattr(d, "page_content", "") or ""

        doc_id = _make_doc_id(agent.id, md, i)
        if doc_id in seen_ids:
            doc_id = f"{doc_id}#dup{i}"
        seen_ids.add(doc_id)

        last_modified = md.get("last_modified_time", "") or md.get("modified_time", "")

        # 3. SKIP if unchanged
        if doc_id in indexed and indexed[doc_id] == last_modified:
            continue

        # 4. Otherwise → new or updated
        results.append({
            "id": doc_id,
            "source": md.get("source", ""),
            "title": md.get("title", ""),
            "page": md.get("page"),
            "slide": md.get("slide"),
            "last_modified": last_modified,
            "text": text,
        })

        # 5. Update state for this file
        indexed[doc_id] = last_modified

    # Save updated state
    agent.index_state = indexed

    return results



# -------- Indexing & Retrieval --------
def _trim_for_embedding(text: str, limit: int = 12000) -> str:
    # keep well under typical embedding limits; adjust if you know your model's exact cap
    t = (text or "").strip()
    return t[:limit] if len(t) > limit else t


from .models import AgentFile
from .db import SessionLocal

# def reindex_agent(agent, _creds_path_ignored: str = ""):
#     """
#     Incremental reindex:
#       - Only embed new or updated docs (by last_modified)
#       - Remove DB/vector entries for files removed from Drive
#       - Return (added, updated_count_placeholder, total_changed)
#     """
#     docs = load_drive_docs(agent)
#     client = get_vector_client()
#     col = ensure_collection(agent, client)
#     provider = provider_from(agent)
#     db = SessionLocal()

#     # -------------------------
#     # Load existing indexed files from DB
#     # -------------------------
#     existing = {}  # file_id -> last_modified
#     for f in db.query(AgentFile).filter_by(agent_id=agent.id):
#         existing[f.file_id] = f.last_modified or ""
#     print(f"[DEBUG] Existing files in DB ({len(existing)}): {list(existing.keys())}")

#     # -------------------------
#     # Build mapping of current drive docs
#     # -------------------------
#     current_ids = []
#     docs_by_id = {}
#     for i, d in enumerate(docs):
#         fid = d["id"]
#         current_ids.append(fid)
#         docs_by_id[fid] = d
#     print(f"[DEBUG] Current files on Drive ({len(current_ids)}): {current_ids}")

#     # -------------------------
#     # Detect deletions: files that existed but are missing from Drive results
#     # -------------------------
#     to_delete = [fid for fid in existing.keys() if fid not in current_ids]
#     if to_delete:
#         try:
#             print(f"[reindex_agent] Deleting {len(to_delete)} removed files: {to_delete}")
#             # Delete metadata rows
#             db.query(AgentFile).filter(AgentFile.agent_id == agent.id, AgentFile.file_id.in_(to_delete)).delete(synchronize_session=False)
#             db.commit()
#         except Exception as e:
#             print(f"[reindex_agent] Error deleting metadata for removed files: {e}")
#         # Attempt to delete from vector store (some Chroma clients support delete)
#         try:
#             col.delete(ids=to_delete)
#         except Exception as e:
#             # not fatal; collection may not support delete() or different API
#             print(f"[reindex_agent] Warning: could not delete ids from collection: {e}")

#     # -------------------------
#     # Decide which docs need embedding (new or updated)
#     # -------------------------
#     to_embed = []     # tuples (file_id, meta, text)
#     to_upsert_meta = []  # list of docs for DB merge
#     dropped_empty = 0

#     for fid in current_ids:
#         d = docs_by_id[fid]
#         text = _trim_for_embedding(d.get("text") or "")
#         if not text:
#             dropped_empty += 1
#             continue

#         drive_last_modified = d.get("last_modified") or ""

#         # If exists and not changed -> skip
#         if fid in existing and drive_last_modified <= (existing.get(fid) or ""):
#             continue

#         # Else: new or updated -> prepare to embed and update DB metadata
#         meta = _sanitize_meta({
#             "title": d.get("title") or "",
#             "source": d.get("source") or "",
#             "page": d.get("page"),
#             "slide": d.get("slide"),
#             "last_modified": drive_last_modified,
#         })

#         to_embed.append((fid, meta, text))
#         to_upsert_meta.append(d)

#     if dropped_empty:
#         print(f"[reindex_agent] Skipped {dropped_empty} empty chunks out of {len(docs)}")

#     if not to_embed and not to_delete:
#         print("[reindex_agent] Nothing new, updated, or deleted.")
#         return (0, 0, 0)

#     # -------------------------
#     # Upsert metadata for new/updated files
#     # -------------------------
#     for d in to_upsert_meta:
#         try:
#             db.merge(AgentFile(
#                 agent_id=agent.id,
#                 file_id=d["id"],
#                 title=d.get("title") or "(Untitled)",
#                 source=d.get("source") or "",
#                 last_modified=d.get("last_modified") or "",
#                 page=d.get("page"),
#                 slide=d.get("slide"),
#             ))
#         except Exception as e:
#             print(f"[reindex_agent] Warning: failed to merge metadata for {d.get('id')}: {e}")
#     db.commit()

#     # -------------------------
#     # Embed & upsert only changed docs (batch + per-item fallback)
#     # -------------------------
#     BATCH = 64
#     added = 0
#     salvaged_total = 0
#     dims = None

#     for i in range(0, len(to_embed), BATCH):
#         batch = to_embed[i:i+BATCH]
#         ids_batch   = [t[0] for t in batch]
#         metas_batch = [t[1] for t in batch]
#         docs_batch  = [t[2] for t in batch]

#         embs_batch = None
#         try:
#             embs_batch = _as_2d_floats(provider.embed(docs_batch))
#         except Exception as e:
#             print(f"[reindex_agent] Batch embed error at [{i}:{i+len(batch)}]: {e}")
#             return (0, 0, 0) 

#         # Salvage per item if needed
#         need_salvage = (
#             embs_batch is None or
#             not isinstance(embs_batch, (list, tuple)) or
#             len(embs_batch) != len(docs_batch)
#         )
#         if not need_salvage and embs_batch:
#             d0 = len(embs_batch[0])
#             if any(len(v) != d0 for v in embs_batch):
#                 need_salvage = True

#         if need_salvage:
#             keep_idx, fixed_embs = [], []
#             for j, txt in enumerate(docs_batch):
#                 try:
#                     one = provider.embed([txt])
#                     one = _as_2d_floats(one)[0]
#                     if dims is None:
#                         dims = len(one)
#                     if len(one) != dims:
#                         print(f"[reindex_agent] Dropped item {i+j}: dim {len(one)} != {dims}")
#                         continue
#                     fixed_embs.append(one)
#                     keep_idx.append(j)
#                 except Exception as ee:
#                     print(f"[reindex_agent] Dropped item {i+j}: {ee}")
#             if not keep_idx:
#                 continue
#             ids_batch   = [ids_batch[j] for j in keep_idx]
#             metas_batch = [metas_batch[j] for j in keep_idx]
#             docs_batch  = [docs_batch[j] for j in keep_idx]
#             embs_batch  = fixed_embs
#             salvaged_total += (len(batch) - len(keep_idx))
#         else:
#             dims = dims or len(embs_batch[0])

#         # Upsert in collection
#         try:
#             col.upsert(ids=ids_batch, documents=docs_batch, metadatas=metas_batch, embeddings=embs_batch)
#             added += len(ids_batch)
#         except Exception as e:
#             print(f"[reindex_agent] Error upserting batch [{i}:{i+len(batch)}]: {e}")

#     if salvaged_total:
#         print(f"[reindex_agent] Salvaged {salvaged_total} items via per-item embedding fallback")

#     total_changed = added + (len(to_delete) if to_delete else 0)
#     print(f"[reindex_agent] Done. Added/Updated embeddings: {added}; Deleted: {len(to_delete)}; Total changed: {total_changed}")
#     return (added, 0, total_changed)


def reindex_agent(agent, _creds_path_ignored: str = ""):
    """
    Incremental reindex using RAW Google Drive API:
      - Calls load_drive_docs_r() to list current Drive files
      - Extracts text using _extract_text_for_doc()
      - Embeds only new/updated files
      - Removes deleted files from DB + vector store
      - Upserts metadata + embeddings
    """

    # -------------------------
    # 1. Load current Drive docs (RAW API)
    # -------------------------
    docs = load_drive_docs_r(agent)
    print("Documents retrieved : " , docs)
    client = get_vector_client()
    col = ensure_collection(agent, client)
    provider = provider_from(agent)
    db = SessionLocal()

    # -------------------------
    # 2. Load existing DB index
    # -------------------------
    existing = {}
    for f in db.query(AgentFile).filter_by(agent_id=agent.id):
        existing[f.file_id] = f.last_modified or ""

    print(f"[DEBUG] Existing files in DB ({len(existing)}): {list(existing.keys())}")

    # -------------------------
    # 3. Map current docs
    # -------------------------
    docs_by_id = {d["id"]: d for d in docs}
    #print("docs_is" +docs_by_id)
    current_ids = list(docs_by_id.keys())

    print(f"[DEBUG] Current files on Drive ({len(current_ids)}): {current_ids}")

    # -------------------------
    # 4. Detect deletions
    # -------------------------
    to_delete = [fid for fid in existing if fid not in current_ids]

    if to_delete:
        print(f"[reindex_agent] Deleting {len(to_delete)} removed files: {to_delete}")
        try:
            db.query(AgentFile).filter(
                AgentFile.agent_id == agent.id,
                AgentFile.file_id.in_(to_delete)
            ).delete(synchronize_session=False)
            db.commit()
        except Exception as e:
            print(f"[reindex_agent] Error deleting metadata: {e}")

        try:
            col.delete(ids=to_delete)
        except Exception as e:
            print(f"[reindex_agent] Warning: could not delete from vector store: {e}")

    # -------------------------
    # 5. Extract text for each Drive doc BEFORE comparing timestamps
    # -------------------------
    for d in docs:
        try:
            text = _extract_text_for_doc(d, agent.owner_id)
            d["text"] = _trim_for_embedding(text)
        except Exception as e:
            print(f"[reindex_agent] Text extraction failed for {d['id']}: {e}")
            d["text"] = ""

    # -------------------------
    # 6. Decide which docs need embedding
    # -------------------------
    to_embed = []
    to_upsert_meta = []
    dropped_empty = 0

    for fid, d in docs_by_id.items():
        text = d.get("text") or ""
        if not text.strip():
            dropped_empty += 1
            continue

        drive_mod = d.get("last_modified") or ""

        # If exists AND has not changed → skip
        if fid in existing and drive_mod <= existing[fid]:
            continue

        # Needs embedding (new or updated)
        meta = _sanitize_meta({
            "title": d.get("title") or "",
            "source": d.get("source") or "",
            "page": d.get("page"),
            "slide": d.get("slide"),
            "last_modified": drive_mod,
        })

        to_embed.append((fid, meta, text))
        to_upsert_meta.append(d)

    if dropped_empty:
        print(f"[reindex_agent] Dropped {dropped_empty} empty docs out of {len(docs)}")

    if not to_embed and not to_delete:
        print("[reindex_agent] Nothing new or changed.")
        return (0, 0, 0)

    # -------------------------
    # 7. Upsert metadata into DB
    # -------------------------
    for d in to_upsert_meta:
        try:
            db.merge(AgentFile(
                agent_id=agent.id,
                file_id=d["id"],
                title=d.get("title") or "(Untitled)",
                source=d.get("source") or "",
                last_modified=d.get("last_modified") or "",
                page=d.get("page"),
                slide=d.get("slide"),
            ))
        except Exception as e:
            print(f"[reindex_agent] Warning: metadata merge failed for {d['id']}: {e}")
    db.commit()

    # -------------------------
    # 8. Batch embedding + fallback salvage
    # -------------------------
    BATCH = 64
    added = 0
    salvaged_total = 0
    dims = None

    for i in range(0, len(to_embed), BATCH):
        batch = to_embed[i:i+BATCH]
        ids_batch   = [t[0] for t in batch]
        metas_batch = [t[1] for t in batch]
        texts_batch = [t[2] for t in batch]

        # Try batch embedding
        try:
            embs_batch = _as_2d_floats(provider.embed(texts_batch))
        except Exception as e:
            print(f"[reindex_agent] Batch embed error {i}-{i+len(batch)}: {e}")
            embs_batch = None

        # ------------- salvage if dimension mismatch or error -------------
        need_salvage = (
            embs_batch is None or
            len(embs_batch) != len(texts_batch)
        )

        if not need_salvage:
            d0 = len(embs_batch[0])
            if any(len(v) != d0 for v in embs_batch):
                need_salvage = True

        if need_salvage:
            print("[reindex_agent] Salvaging batch...")
            fixed_embs, keep_idx = [], []

            for j, txt in enumerate(texts_batch):
                try:
                    one = _as_2d_floats(provider.embed([txt]))[0]
                    if dims is None:
                        dims = len(one)
                    if len(one) != dims:
                        print(f"[reindex_agent] Drop item {ids_batch[j]}: dim mismatch")
                        continue
                    keep_idx.append(j)
                    fixed_embs.append(one)
                except Exception as e:
                    print(f"[reindex_agent] Drop item {ids_batch[j]}: {e}")

            if not keep_idx:
                continue

            ids_batch   = [ids_batch[j] for j in keep_idx]
            metas_batch = [metas_batch[j] for j in keep_idx]
            texts_batch = [texts_batch[j] for j in keep_idx]
            embs_batch  = fixed_embs
            salvaged_total += (len(batch) - len(keep_idx))
        else:
            dims = dims or len(embs_batch[0])

        # -------------------------
        # Upsert embeddings
        # -------------------------
        try:
            col.upsert(
                ids=ids_batch,
                documents=texts_batch,
                metadatas=metas_batch,
                embeddings=embs_batch
            )
            added += len(ids_batch)
        except Exception as e:
            print(f"[reindex_agent] Error upserting batch: {e}")

    if salvaged_total:
        print(f"[reindex_agent] Salvaged {salvaged_total} items")

    total_changed = added + len(to_delete)
    print(f"[reindex_agent] Done. Added: {added}, Deleted: {len(to_delete)}, Total changed: {total_changed}")

    return (added, 0, total_changed)








