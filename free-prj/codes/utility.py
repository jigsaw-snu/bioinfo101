# Errors
class IndexNotMatching(Exception):
    def __init__(self):
        super().__init__("Index is Not Matching Correctly!")