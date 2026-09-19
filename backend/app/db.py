"""Database engine setup. Defaults to a local, zero-setup SQLite file (via
the async aiosqlite driver) so persistence works out of the box with no
external service to run — the same on-disk-by-default philosophy already
used for Chroma's .chroma_data. Point DATABASE_URL at a real Postgres
instance (e.g. "postgresql+asyncpg://user:pass@host/db") to swap it in;
nothing else in the persistence layer changes, since it's written against
plain SQLAlchemy Core/ORM, not anything SQLite-specific."""

import os
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

_DEFAULT_SQLITE_PATH = Path(__file__).parent.parent / "app_data.db"
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite+aiosqlite:///{_DEFAULT_SQLITE_PATH}")


class Base(DeclarativeBase):
    pass


_engine: AsyncEngine = create_async_engine(DATABASE_URL, echo=False)
_session_factory = async_sessionmaker(_engine, expire_on_commit=False)


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    return _session_factory


async def init_db() -> None:
    """Creates any missing tables directly from the current ORM models.
    Safe to call on every server startup for local/zero-setup dev — does
    nothing to tables that already exist, and requires no separate command.

    This is NOT how schema changes should reach a real deployment, though:
    `create_all` only ever adds missing tables, it never alters an existing
    one (add a column, change a type, etc.), so a schema change made after
    the first deploy would silently do nothing here. Real deployments
    should run `alembic upgrade head` (see backend/alembic/) instead, which
    tracks and applies versioned migrations — `alembic revision
    --autogenerate -m "..."` after changing a model in persistence.py,
    review the generated migration, then apply it."""
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
