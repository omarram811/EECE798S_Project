# app/rag.py
from typing import List, Tuple, Dict
import os, json, shutil, logging
from pathlib import Path

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
    """Load Google Drive docs (recursive) and normalize."""
    folder_id = file_id_from_url_or_id(agent.drive_folder_id or "")
    auth_kwargs = _build_drive_auth_kwargs(agent.owner_id)
    docs_raw = _load_drive_docs_raw(folder_id, auth_kwargs)

    results = []
    seen_ids = set()
    for i, d in enumerate(docs_raw):
        md = getattr(d, "metadata", {}) or {}
        text = getattr(d, "page_content", "") or ""

        doc_id = _make_doc_id(agent.id, md, i)
        # Guard against any accidental dup even after suffixing
        if doc_id in seen_ids:
            doc_id = f"{doc_id}#dup{i}"
        seen_ids.add(doc_id)

        results.append({
            "id": doc_id,
            "source": md.get("source", ""),
            "title": md.get("title", ""),
            "page": md.get("page"),
            "slide": md.get("slide"),
            "last_modified": md.get("last_modified_time", "") or md.get("modified_time", ""),
            "text": text,
        })
    return results


# -------- Indexing & Retrieval --------
def _trim_for_embedding(text: str, limit: int = 12000) -> str:
    # keep well under typical embedding limits; adjust if you know your model's exact cap
    t = (text or "").strip()
    return t[:limit] if len(t) > limit else t

def reindex_agent(agent, _creds_path_ignored: str = "") -> Tuple[int, int, int]:
    docs = load_drive_docs(agent)
    client = get_vector_client()
    col = ensure_collection(agent, client)
    provider = provider_from(agent)

    # 1) Build triples
    triples = []
    dropped_empty = 0
    for d in docs:
        text = _trim_for_embedding(d.get("text") or "")
        if not text:
            dropped_empty += 1
            continue
        meta = _sanitize_meta({
            "title": d.get("title") or "",
            "source": d.get("source") or "",
            "page": d.get("page"),
            "slide": d.get("slide"),
            "last_modified": d.get("last_modified") or "",
        })
        triples.append((d["id"], meta, text))

    if dropped_empty:
        print(f"[reindex_agent] Skipped {dropped_empty} empty chunks out of {len(docs)}")
    if not triples:
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
        if not need_salvage and embs_batch:
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

    return (added, 0, added)


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


