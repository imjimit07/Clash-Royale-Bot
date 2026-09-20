"""Card cycle tracking (Part 3.1)."""

from __future__ import annotations


class CardTracker:
    """Track 4-card hand + next-card slide-in across an 8-card deck."""

    def __init__(self, all_cards: list[str] | tuple[str, ...]) -> None:
        self.hand: list[str] = []
        self.next_card: str | None = None
        self.cycle: list[str] = list(all_cards)
        self.played: list[str] = []

    def update_from_hand(self, detected_hand: list[str]) -> None:
        self.hand = list(detected_hand)

    def play(self, card: str) -> None:
        if card in self.hand:
            self.hand.remove(card)
            self.played.append(card)
            if self.next_card:
                self.hand.append(self.next_card)
                self.next_card = None

    def predict_next(self) -> str | None:
        return self.next_card
