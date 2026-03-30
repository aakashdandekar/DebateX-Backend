import os
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta, timezone
from fastapi.security import HTTPBearer
from fastapi import Depends, HTTPException

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

def hash(password: str) -> str:
    return pwd_context.hash(password)

def check_hash(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)

def create_access_token(user_id: str) -> str:
    return jwt.encode({
        "user_id": user_id,
        "exp": datetime.now(tz=timezone.utc) + timedelta(hours=2)
    }, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token=Depends(security)) -> str:
    try:
        payload = jwt.decode(
            token.credentials,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        return payload["user_id"]

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")