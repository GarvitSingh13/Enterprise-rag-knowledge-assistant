"""Authentication Service for Enterprise RAG.

Provides cryptographically secure PBKDF2 password hashing,
user credential verification, and signed JSON Web Token (JWT) management.
"""

from typing import Dict, Optional, List
import hashlib
import hmac
import secrets
from datetime import datetime, timezone, timedelta
import jwt
from enterprise_rag.core.models import Department
from enterprise_rag.security.models import User, UserRole, TokenPayload, UserPublicProfile


class AuthService:
    """Manages user authentication and JWT token lifecycle."""

    DEFAULT_SECRET_KEY = "enterprise-rag-super-secret-production-signing-key-2026"
    ALGORITHM = "HS256"
    TOKEN_EXPIRY_HOURS = 8

    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or self.DEFAULT_SECRET_KEY
        self.users_db: Dict[str, User] = {}
        self._seed_default_enterprise_users()

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
        """Hash a plaintext password using PBKDF2-HMAC-SHA256 with 100,000 iterations."""
        salt = salt or secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            iterations=100_000,
        ).hex()
        return hashed, salt

    @classmethod
    def verify_password(cls, password: str, salt: str, expected_hash: str) -> bool:
        """Verify password against expected hash using constant-time comparison."""
        computed_hash, _ = cls.hash_password(password, salt)
        return hmac.compare_digest(computed_hash, expected_hash)

    def register_user(
        self,
        username: str,
        password: str,
        full_name: str,
        department: Department,
        roles: List[UserRole],
    ) -> User:
        """Register a new user with hashed credentials."""
        clean_user = username.lower().strip()
        if clean_user in self.users_db:
            raise ValueError(f"User '{clean_user}' already exists.")

        hashed_pw, salt = self.hash_password(password)
        user = User(
            username=clean_user,
            full_name=full_name,
            department=department,
            roles=roles,
            hashed_password=hashed_pw,
            salt=salt,
        )
        self.users_db[clean_user] = user
        return user

    def _seed_default_enterprise_users(self) -> None:
        """Seed realistic enterprise personas across departments and clearance tiers."""
        seed_users = [
            ("alice_emp", "EmployeePass123!", "Alice Johnson", Department.HR, [UserRole.EMPLOYEE]),
            (
                "bob_finance",
                "FinancePass123!",
                "Bob Martinez",
                Department.FINANCE,
                [UserRole.EMPLOYEE, UserRole.FINANCE_ANALYST],
            ),
            (
                "david_dev",
                "DevPass123!",
                "David Kim",
                Department.ENGINEERING,
                [UserRole.EMPLOYEE, UserRole.ENGINEER],
            ),
            (
                "sarah_ceo",
                "ExecutivePass123!",
                "Sarah Connor",
                Department.EXECUTIVE,
                [UserRole.EMPLOYEE, UserRole.C_SUITE, UserRole.ADMIN],
            ),
        ]
        for uname, pwd, name, dept, roles in seed_users:
            self.register_user(uname, pwd, name, dept, roles)

    def login(self, username: str, password: str) -> str:
        """
        Authenticate user credentials and return a signed JWT access token.
        """
        clean_user = username.lower().strip()
        user = self.users_db.get(clean_user)
        if not user or not self.verify_password(password, user.salt, user.hashed_password):
            raise ValueError("Invalid username or password.")

        return self.create_access_token(user)

    def create_access_token(self, user: User) -> str:
        """Generate a signed JWT token with claims and expiry."""
        now = datetime.now(timezone.utc)
        exp = now + timedelta(hours=self.TOKEN_EXPIRY_HOURS)

        claims = {
            "sub": user.user_id,
            "username": user.username,
            "full_name": user.full_name,
            "department": user.department.value,
            "roles": [r.value for r in user.roles],
            "iat": int(now.timestamp()),
            "exp": int(exp.timestamp()),
        }
        return jwt.encode(claims, self.secret_key, algorithm=self.ALGORITHM)

    def verify_token(self, token: str) -> TokenPayload:
        """
        Verify, decode, and validate a JWT access token.
        Raises ValueError if invalid, tampered, or expired.
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.ALGORITHM])
            return TokenPayload(**payload)
        except jwt.ExpiredSignatureError:
            raise ValueError("Access token has expired. Please log in again.")
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid or tampered token: {e}")

    def get_public_profile(self, username: str) -> Optional[UserPublicProfile]:
        """Fetch sanitized public user profile."""
        u = self.users_db.get(username.lower().strip())
        if not u:
            return None
        return UserPublicProfile(
            user_id=u.user_id,
            username=u.username,
            full_name=u.full_name,
            department=u.department,
            roles=u.roles,
        )
