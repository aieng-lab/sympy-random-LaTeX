from .expr import Basic

class String:
    def __init__(self, text):
        self.text = text

    @property
    def free_symbols(self):
        return set()

    def __str__(self):
        return self.text

class Text(Basic):
    pass