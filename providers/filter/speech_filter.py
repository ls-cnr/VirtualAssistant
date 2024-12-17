import re
from ..base import TextFilterProvider

class SpeechFilter(TextFilterProvider):
    def filter(self, text: str) -> str:
        """Rimuove elementi non pronunciabili dal testo."""

        # Lista di pattern da rimuovere
        patterns = [
            # Emoji Unicode (range esteso)
            r'[\U00010000-\U0010ffff]',

            # Emoticon comuni
            r'[:;=]-?[)(/\\|dpDP]',  # :) :-) ;) ;-) =) :-D :p ecc.
            r'[xX]-?[dD]',           # xD X-D
            r':[oO]',                # :o :O
            r'\^[_-]?\^',           # ^^ ^_^ ^-^
            r'>?[_-]?<',            # >< >_< >-
            r'[oO][._][oO]',        # o.o O.O

            # Testo tra asterischi (azioni/emozioni)
            r'\*[^*]+\*',

            # Emoji testuali
            r':[a-zA-Z_]+:',

            # Emoji testuali comuni
            r'<3',                   # cuore
            r'</3',                  # cuore spezzato
            r'\(y\)',                # thumbs up
            r'\(n\)',                # thumbs down

            # Pulizia spazi multipli
            r'\s+'
        ]

        # Applica tutti i pattern
        for pattern in patterns:
            text = re.sub(pattern, ' ', text)

        # Pulisci spazi in eccesso e trimma
        text = ' '.join(text.split())
        return text.strip()

    def print_filtered(self, text: str) -> None:
        """Stampa il testo prima e dopo il filtraggio per debug"""
        filtered = self.filter(text)
        print("Original:", text)
        print("Filtered:", filtered)
