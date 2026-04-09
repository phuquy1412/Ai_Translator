"""
Context Memory - Lưu ngữ cảnh dịch giữa các trang
"""
from collections import deque


class ContextMemory:
    def __init__(self, max_lines=20):
        self.history = deque(maxlen=max_lines)  # Tự động xóa cũ
        self.characters = {}   # {tên_gốc: tên_việt}
        self.terms = {}        # {thuật_ngữ_gốc: thuật_ngữ_việt}

    def add(self, original: str, translated: str):
        """Thêm 1 câu vào context"""
        self.history.append(f"[{original}] → [{translated}]")

    def add_character(self, original: str, translated: str):
        """Thêm tên nhân vật"""
        self.characters[original] = translated

    def add_term(self, original: str, translated: str):
        """Thêm thuật ngữ đặc biệt (tên chiêu, địa danh...)"""
        self.terms[original] = translated

    def get_prompt(self) -> str:
        """Tạo context string để đưa vào prompt"""
        sections = []

        if self.characters:
            chars = "\n".join(f"  {k} → {v}" for k, v in self.characters.items())
            sections.append(f"Tên nhân vật:\n{chars}")

        if self.terms:
            terms = "\n".join(f"  {k} → {v}" for k, v in self.terms.items())
            sections.append(f"Thuật ngữ đặc biệt:\n{terms}")

        if self.history:
            recent = "\n".join(list(self.history)[-10:])
            sections.append(f"Câu thoại gần đây:\n{recent}")

        if not sections:
            return ""

        return "=== NGỮ CẢNH ===\n" + "\n\n".join(sections) + "\n================\n\n"

    def clear(self):
        """Reset khi bắt đầu truyện mới"""
        self.history.clear()
        self.characters.clear()
        self.terms.clear()

    def __repr__(self):
        return f"ContextMemory(history={len(self.history)}, chars={len(self.characters)}, terms={len(self.terms)})"