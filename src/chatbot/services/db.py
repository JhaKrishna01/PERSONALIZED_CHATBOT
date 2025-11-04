"""SQLite persistence layer for conversation turns."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Iterable, List

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATABASE_PATH = PROJECT_ROOT / "chatbot.db"


class Base(DeclarativeBase):
    pass


class ConversationTurn(Base):
    """ORM model for storing user and assistant dialogue turns."""

    __tablename__ = "conversation_turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    user_message: Mapped[str] = mapped_column(Text)
    bot_reply: Mapped[str] = mapped_column(Text)
    emotions: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


engine = create_engine(f"sqlite:///{DATABASE_PATH}", echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, class_=Session)
Base.metadata.create_all(bind=engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""

    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def save_turn(user_id: str, message: str, reply: str, emotions: Iterable[str]) -> None:
    """Persist a single dialogue turn to the database."""

    serialized_emotions = ",".join(emotions)
    with get_session() as session:
        turn = ConversationTurn(
            user_id=user_id,
            user_message=message,
            bot_reply=reply,
            emotions=serialized_emotions,
        )
        session.add(turn)


def fetch_history(user_id: str, max_turns: int = 10) -> List[ConversationTurn]:
    """Fetch the most recent turns for a user, newest first."""

    with get_session() as session:
        results = (
            session.query(ConversationTurn)
            .filter(ConversationTurn.user_id == user_id)
            .order_by(ConversationTurn.created_at.desc())
            .limit(max_turns)
            .all()
        )
    return results