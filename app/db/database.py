"""Engine and session handling for SQLite."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base

logger = logging.getLogger(__name__)


class Database:
    """Owns the engine. ``path=None`` gives a private in-memory database (tests)."""

    def __init__(self, path: Optional[Path] = None) -> None:
        if path is None:
            self.engine: Engine = create_engine(
                "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.engine = create_engine(f"sqlite:///{path.as_posix()}",
                                        connect_args={"check_same_thread": False})

        @event.listens_for(self.engine, "connect")
        def _pragmas(dbapi_conn, _record) -> None:  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL" if path is not None else "PRAGMA journal_mode=MEMORY")
            cur.close()

        self._factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def init(self) -> None:
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Transactional scope: commit on success, roll back on any error."""
        db = self._factory()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def ping(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            logger.exception("database health check failed")
            return False

    def dispose(self) -> None:
        self.engine.dispose()
