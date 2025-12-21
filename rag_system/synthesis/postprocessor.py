# rag_system/synthesis/postprocessor.py

import re
from typing import List, Tuple

def remove_repetition(text: str) -> str:
    """Remove repetitive sentences or phrases"""
    sentences = text.split('. ')
    if len(sentences) < 2:
        return text
    
    seen = set()
    unique_sentences = []
    for sentence in sentences:
        # Normalize sentence for comparison
        normalized = sentence.lower().strip()
        # Skip if very similar to a previous sentence
        is_duplicate = False
        for seen_sent in seen:
            # Check if sentences are very similar (80% overlap)
            if normalized and seen_sent:
                words1 = set(normalized.split())
                words2 = set(seen_sent.split())
                if words1 and words2:
                    overlap = len(words1 & words2) / len(words1 | words2)
                    if overlap > 0.8:
                        is_duplicate = True
                        break
        
        if not is_duplicate and normalized:
            unique_sentences.append(sentence)
            seen.add(normalized)
    
    return '. '.join(unique_sentences) + ('.' if text.endswith('.') else '')

def extract_lists(text: str) -> str:
    """Improve list formatting"""
    # Look for numbered or bulleted lists and ensure proper formatting
    lines = text.split('\n')
    formatted_lines = []
    in_list = False
    
    for line in lines:
        stripped = line.strip()
        # Detect list items
        if re.match(r'^\d+[\.\)]\s', stripped) or stripped.startswith('- ') or stripped.startswith('* '):
            if not in_list:
                formatted_lines.append('')
            in_list = True
            formatted_lines.append(stripped)
        else:
            if in_list and stripped:
                formatted_lines.append('')
            in_list = False
            if stripped:
                formatted_lines.append(stripped)
    
    return '\n'.join(formatted_lines)

def remove_meta_phrases(text: str) -> str:
    """Remove meta-phrases that indicate the model is explaining how to answer"""
    meta_phrases = [
        r'based on the provided documents,?\s*',
        r'in response to the user query[,\s]*',
        r'my answer would be\s*',
        r'to answer this question[,\s]*',
        r'according to the documents[,\s]*',
        r'based on the information provided[,\s]*',
        r'the documents (?:do not |don\'t )?contain',
        r'to provide a comprehensive answer[,\s]*',
        r'to address the specific aspect[,\s]*',
        r'to cite sources[,\s]*',
        r'if the documents don\'t contain[,\s]*',
        r'to ensure accuracy[,\s]*',
    ]
    
    cleaned = text
    for phrase in meta_phrases:
        cleaned = re.sub(phrase, '', cleaned, flags=re.IGNORECASE)
    
    return cleaned.strip()

def clean_answer(answer: str) -> str:
    """Main post-processing function to clean and format answers"""
    if not answer or len(answer.strip()) < 10:
        return answer
    
    # Remove meta-phrases
    cleaned = remove_meta_phrases(answer)
    
    # Remove repetition
    cleaned = remove_repetition(cleaned)
    
    # Improve list formatting
    cleaned = extract_lists(cleaned)
    
    # Fix capitalization at the start
    cleaned = cleaned.strip()
    if cleaned and cleaned[0].islower():
        # Capitalize first letter if it's lowercase
        cleaned = cleaned[0].upper() + cleaned[1:]
    
    # Remove excessive whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
    
    # Remove leading/trailing whitespace
    cleaned = cleaned.strip()
    
    # Ensure it ends with punctuation if it's a sentence
    if cleaned and not cleaned[-1] in '.!?':
        # Check if it looks like a complete sentence
        if len(cleaned) > 50 and not cleaned.endswith(':'):
            cleaned += '.'
    
    return cleaned

def detect_incomplete_answer(answer: str, query: str) -> Tuple[bool, str]:
    """Detect if answer appears incomplete and needs continuation"""
    if not answer or len(answer.strip()) < 20:
        return True, "Answer is too short"
    
    # Check if answer ends mid-sentence (likely cut off)
    answer_trimmed = answer.strip()
    if answer_trimmed:
        last_char = answer_trimmed[-1]
        # If doesn't end with punctuation, might be incomplete
        if last_char not in '.!?;:':
            # Check if last sentence is very short (likely cut off)
            last_sentence = answer_trimmed.split('.')[-1].strip()
            if len(last_sentence) > 0 and len(last_sentence) < 20:
                return True, "Answer appears cut off mid-sentence"
        
        # Check for incomplete list patterns
        if re.search(r'\d+\.\s*$', answer_trimmed) or re.search(r'-\s*$', answer_trimmed):
            return True, "Answer appears to have incomplete list"
        
        # Check for trailing "and", "or", "but" (likely incomplete)
        if re.search(r'\s+(and|or|but|also|furthermore|additionally)\s*$', answer_trimmed, re.IGNORECASE):
            return True, "Answer appears to have incomplete thought"
    
    # Check for common "I don't know" patterns (these are complete, just indicate missing info)
    uncertainty_patterns = [
        r"cannot (?:fully )?answer",
        r"don't (?:have|contain) (?:enough|sufficient)",
        r"not (?:enough|sufficient) information",
        r"unable to (?:fully )?answer",
    ]
    
    for pattern in uncertainty_patterns:
        if re.search(pattern, answer, re.IGNORECASE):
            return False, "Answer indicates missing information (complete but limited)"
    
    # For list queries, check if answer seems incomplete
    if any(word in query.lower() for word in ['list', 'what are', 'chapters', 'topics']):
        # Check if we have numbered items but they seem incomplete
        numbered_items = re.findall(r'\d+\.', answer)
        if numbered_items and len(numbered_items) < 3:
            # Might need more items
            return True, "List query may have incomplete items"
    
    # Check if answer is too generic
    generic_phrases = [
        "based on the documents",
        "the documents provide",
        "according to the information",
    ]
    
    generic_count = sum(1 for phrase in generic_phrases if phrase.lower() in answer.lower())
    if generic_count > 2 and len(answer) < 100:
        return True, "Answer seems too generic"
    
    return False, "Answer appears complete"

def validate_answer_completeness(answer: str, query: str) -> Tuple[bool, str]:
    """Validate if answer seems complete (wrapper for backward compatibility)"""
    is_incomplete, reason = detect_incomplete_answer(answer, query)
    return not is_incomplete, reason
