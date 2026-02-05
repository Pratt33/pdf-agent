import hashlib
from collections import OrderedDict

class PDFCache:
    def __init__(self, max_size=2):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get_hash(self, pdf_path):
        """Generate hash for PDF file"""
        with open(pdf_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def get(self, pdf_hash):
        """Retrieve cached data"""
        if pdf_hash in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(pdf_hash)
            return self.cache[pdf_hash]
        return None
    
    def set(self, pdf_hash, data):
        """Store data in cache"""
        if pdf_hash in self.cache:
            self.cache.move_to_end(pdf_hash)
        else:
            self.cache[pdf_hash] = data
            # Remove oldest if over limit
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()

# Global cache instance
pdf_cache = PDFCache(max_size=10)
