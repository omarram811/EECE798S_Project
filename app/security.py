# app/security.py
import os
from itsdangerous import TimestampSigner, BadSignature
from fastapi import Request, Response
from passlib.context import CryptContext

# Password hashing (no bcrypt cap/issues)
_pwd = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

def hash_password(password: str) -> str:
    return _pwd.hash(password)

def verify_password(password: str, password_hash: str) -> bool:
    return _pwd.verify(password, password_hash)

# Use a DIFFERENT cookie name than Starlette's default "session"
COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "ta_session")

def _get_signer() -> TimestampSigner:
    # Read SECRET_KEY at call time (works across reloads)
    return TimestampSigner(os.getenv("SECRET_KEY", "dev"))

def set_session_cookie(resp: Response, user_id: int) -> None:
    token = _get_signer().sign(str(user_id)).decode("utf-8")
    # path=/ so all routes see it; SameSite=Lax is fine for this app
    resp.set_cookie(COOKIE_NAME, token, httponly=True, samesite="lax", path="/")

def clear_session_cookie(resp: Response) -> None:
    resp.delete_cookie(COOKIE_NAME, path="/")

def get_current_user_id(request: Request):
    raw = request.cookies.get(COOKIE_NAME)
    if not raw:
        return None
    try:
        val = _get_signer().unsign(raw, max_age=60 * 60 * 24 * 30)  # 30 days
        return int(val.decode("utf-8"))
    except BadSignature:
        return None
