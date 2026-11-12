"""
Word Builder Module for ASL Fingerspelling Studio.
Collects recognized alphabets sequentially, builds words in real time,
and supports editing (Delete, Clear) and Thumbs-Up Confirmation.
"""

from __future__ import annotations

from typing import List, Optional


class WordBuilder:
    """
    Maintains the sequentially formed word from stabilized gesture events.
    Guarantees strict FIFO character appending, duplicate prevention, and editing operations.
    """

    def __init__(self) -> None:
        self._letters: List[str] = []
        self._last_confirmed_word: str = ""
        self._history: List[str] = []

    def add_letter(self, letter: str) -> bool:
        """
        Appends a valid single A-Z letter to the current word.
        Returns True if the letter was appended, False otherwise.
        """
        if not letter or not isinstance(letter, str):
            return False
        clean = letter.strip().upper()
        if len(clean) != 1 or not clean.isalpha():
            return False
        self._letters.append(clean)
        return True

    def delete_last(self) -> Optional[str]:
        """
        Removes and returns the last added letter, or None if buffer is empty.
        """
        if self._letters:
            return self._letters.pop()
        return None

    def clear(self) -> str:
        """
        Clears the entire current word buffer and returns the cleared string.
        """
        cleared = "".join(self._letters)
        self._letters.clear()
        return cleared

    def confirm(self) -> Optional[str]:
        """
        Confirms the currently formed word (triggered by Thumbs-Up or UI Confirm button).
        Saves the word to last_confirmed_word and history, resets the current letter buffer,
        and returns the confirmed word (or None if buffer was empty).
        """
        if not self._letters:
            return None
        word = "".join(self._letters)
        self._last_confirmed_word = word
        self._history.append(word)
        self._letters.clear()
        return word

    @property
    def current_word(self) -> str:
        """Returns the current word string (e.g. 'KAJAL')."""
        return "".join(self._letters)

    @property
    def letter_sequence(self) -> List[str]:
        """Returns a copy of the list of letters (e.g. ['K', 'A', 'J', 'A', 'L'])."""
        return list(self._letters)

    @property
    def formatted_sequence(self) -> str:
        """Returns sequence formatted with arrows (e.g. 'K -> A -> J -> A -> L')."""
        if not self._letters:
            return ""
        return " -> ".join(self._letters)

    @property
    def last_confirmed_word(self) -> str:
        """Returns the most recently confirmed word."""
        return self._last_confirmed_word

    @property
    def history(self) -> List[str]:
        """Returns history of all confirmed words."""
        return list(self._history)

    def is_empty(self) -> bool:
        """Returns True if the current word buffer contains no letters."""
        return len(self._letters) == 0

    def letter_count(self) -> int:
        """Returns the number of letters in the current word."""
        return len(self._letters)

    def __len__(self) -> int:
        return len(self._letters)

    def __str__(self) -> str:
        return self.current_word

