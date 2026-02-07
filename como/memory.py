"""Conversation Memory with CoMo filtering."""
from datetime import datetime
from typing import Optional
import json
from pathlib import Path

from ato.adict import ADict

from como.como import CoMo
from como.schemas import Message, MemoryChunk


class ConversationMemory:
    """
    Conversation memory that uses CoMo for query-aware context retrieval.
    
    Key insight: Old conversations have retention (they're important)
    but may have high exclusion for current query (off-topic now).
    """
    
    def __init__(self, config: ADict):
        self._config = config
        self._como = CoMo(config)
        self._history: list[Message] = []
    
    @property
    def history(self) -> list[Message]:
        return self._history
    
    def add(self, role: str, content: str) -> Message:
        """Add a message to conversation history."""
        msg = Message(role=role, content=content, timestamp=datetime.now())
        self._history.append(msg)
        return msg
    
    def add_user(self, content: str) -> Message:
        return self.add('user', content)
    
    def add_assistant(self, content: str) -> Message:
        return self.add('assistant', content)
    
    def _history_to_text(self) -> str:
        """Convert history to text for CoMo processing."""
        lines = []
        for msg in self._history:
            time_str = msg.timestamp.strftime('%Y-%m-%d %H:%M')
            lines.append(f"[{time_str}] {msg.role}: {msg.content}")
        return '\n'.join(lines)
    
    def retrieve(self, query: str) -> list[MemoryChunk]:
        """
        Retrieve relevant conversation context for the given query.
        
        Uses CoMo to filter out messages that are:
        - Low retention (trivial)
        - High exclusion (important but off-topic for this query)
        """
        if not self._history:
            return []
        
        history_text = self._history_to_text()
        return self._como(history_text, query=query)
    
    def score_all(self, query: str) -> list[MemoryChunk]:
        """
        Score ALL conversation chunks without filtering.
        
        Use this to see why certain messages were excluded.
        Returns all chunks with their retention and exclusion scores.
        """
        if not self._history:
            return []
        
        history_text = self._history_to_text()
        return self._como.score_all(history_text, query=query)
    
    def get_context(self, query: str, max_chars: int = 4000) -> str:
        """
        Get filtered context string for LLM consumption.
        
        Returns only the most relevant conversation history
        for the current query.
        """
        chunks = self.retrieve(query)
        
        if not chunks:
            return ""
        
        context_parts = []
        total_chars = 0
        
        for chunk in sorted(chunks, key=lambda c: c.retention_score, reverse=True):
            if total_chars+len(chunk.content) > max_chars:
                break
            context_parts.append(chunk.content)
            total_chars += len(chunk.content)
        
        return '\n'.join(context_parts)
    
    def clear(self):
        """Clear all conversation history."""
        self._history = []
    
    def save(self, path: str):
        """Save conversation history to JSON file."""
        data = [msg.model_dump() for msg in self._history]
        for d in data:
            d['timestamp'] = d['timestamp'].isoformat()
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2))
    
    def load(self, path: str):
        """Load conversation history from JSON file."""
        data = json.loads(Path(path).read_text())
        self._history = []
        for d in data:
            d['timestamp'] = datetime.fromisoformat(d['timestamp'])
            self._history.append(Message(**d))
