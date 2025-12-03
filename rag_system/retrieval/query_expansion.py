# rag_system/retrieval/query_expansion.py

import re
from ingestion.embedder import get_embedding_model
import numpy as np

def expand_query(query: str, num_expansions: int = 2):
    """
    Expand query with synonyms and variations for better recall.
    Returns list of expanded query variations.
    """
    expanded = [query]  # Always include original
    
    try:
        # Simple query rewriting strategies
        # 1. Question to statement conversion
        question_patterns = [
            (r'^what is (.+)\?$', r'\1'),
            (r'^what are (.+)\?$', r'\1'),
            (r'^how does (.+)\?$', r'\1'),
            (r'^how do (.+)\?$', r'\1'),
            (r'^why (.+)\?$', r'\1'),
            (r'^when (.+)\?$', r'\1'),
            (r'^where (.+)\?$', r'\1'),
            (r'^who (.+)\?$', r'\1'),
        ]
        
        for pattern, replacement in question_patterns:
            if re.match(pattern, query.lower()):
                statement = re.sub(pattern, replacement, query.lower(), flags=re.IGNORECASE)
                if statement.strip() and statement != query.lower():
                    expanded.append(statement.strip())
        
        # 2. Add query variations with common synonyms
        # Simple keyword-based expansion
        synonyms_map = {
            'how': ['method', 'way', 'process'],
            'what': ['definition', 'meaning', 'concept'],
            'why': ['reason', 'cause', 'purpose'],
            'explain': ['describe', 'detail', 'clarify'],
            'difference': ['distinction', 'comparison', 'contrast'],
        }
        
        words = query.lower().split()
        for word in words:
            if word in synonyms_map:
                for synonym in synonyms_map[word][:1]:  # Take first synonym
                    variation = query.lower().replace(word, synonym)
                    if variation not in expanded:
                        expanded.append(variation)
        
        # 3. Generate query without stop words (focus on key terms)
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                     'would', 'should', 'could', 'may', 'might', 'must', 'can'}
        key_terms = [w for w in words if w not in stop_words and len(w) > 2]
        if len(key_terms) > 1:
            key_query = ' '.join(key_terms)
            if key_query not in expanded:
                expanded.append(key_query)
        
        # Limit number of expansions
        return expanded[:num_expansions + 1]  # +1 for original
    
    except Exception as e:
        print(f"[Warning] Query expansion failed: {e}")
        return [query]  # Return original on error
