import pylcs
import unicodedata
import cologne_phonetics

from nltk.util import ngrams
from nltk.metrics.distance import edit_distance


def dice(str1, str2, n):
    """
    Calculate XXDICE similarity between two strings
    DICE = 2 * |intersection| / (|str1| + |str2|)
    """
    if not str1 and not str2:
        return 1.0
    if not str1 or not str2:
        return 0.0
    
    # Convert strings to sets of characters (unique characters)
    set1 = set(ngrams(str1, n=n))
    set2 = set(ngrams(str2, n=n))
    
    # Find intersection
    intersection = set1.intersection(set2)
    
    # Calculate DICE
    if len(set1) + len(set2) == 0:
        return 0.0
    
    return 2 * len(intersection) / (len(set1) + len(set2))


def xdice(str1, str2, do_xxdice=False):
  # Convert strings to sets of characters (unique characters)
  set1 = set(trigram[0] + trigram[2] for trigram in ngrams(str1, n=3))
  set2 = set(trigram[0] + trigram[2] for trigram in ngrams(str2, n=3))

  if do_xxdice:
    str1_trigrams_positions = [(trigram[0] + trigram[2], position) for position, trigram in enumerate(ngrams(str1, n=3))]
    str2_trigrams_positions = [(trigram[0] + trigram[2], position) for position, trigram in enumerate(ngrams(str2, n=3))]
    
    done = set()
    factors = []
    for trigram1, position1 in str1_trigrams_positions[::-1]:
      for trigram2, position2 in str2_trigrams_positions[::-1]:
        if trigram1 == trigram2 and trigram1 not in done:
          factor = 1 / (1 + position1 + position2)**2
          factors.append(factor)
          done.add(trigram1)
    len_intersection = sum(factors)
  else:
    # Find intersection
    len_intersection = len(set1.intersection(set2))
    
  # Calculate DICE
  if len(set1) + len(set2) == 0:
      return 0.0
  
  return 2 * len_intersection / (len(set1) + len(set2))


def lcsr(str1, str2):
  return pylcs.lcs_sequence_length(str1, str2) / max(len(str1), len(str2))


def compute_cologne_phonetics(str1, str2):
  s1_normalized, s1_encoded = cologne_phonetics.encode(str1)[0]
  s2_normalized, s2_encoded = cologne_phonetics.encode(str2)[0]
  return edit_distance(str(s1_encoded), str(s2_encoded))


def ngrams_jaccard(string_1, string_2, n=2):
    bigrams_s1 = set(ngrams(string_1, n))
    bigrams_s2 = set(ngrams(string_2, n))
    
    intersection = len(bigrams_s1.intersection(bigrams_s2))
    union = len(bigrams_s1.union(bigrams_s2))

    return intersection / union if union != 0 else 0


def normalized_edit_distance(string_1, string_2):
  return edit_distance(string_1, string_2) / max(len(string_1), len(string_2))


def _id(str1, str2, n):
  str1_ngrams = list(ngrams(str1, n=n))
  str2_ngrams = list(ngrams(str2, n=n))
  return 1/n * sum(a == b for a, b in zip(str1_ngrams, str2_ngrams))


def prefix(string_1, string_2):
  if len(string_1) + len(string_2) == 0:
    return 0
  
  for i in range(min(len(string_1), len(string_2))):
    if string_1[i] != string_2:
      return i / max(len(string_1), len(string_2))
    
  return i


def nsimdist(str1, str2, n, simdist):
  """
  implements BI-SIM and TRI-SIM for n=2 and n=3 (if simdist="sim"), or BI-DIST and TRI-DIST for simdist="dist"
  https://www.site.uottawa.ca/~ofrunza/Page/Papers/ranlp2005_cognates.pdf
  """
  if len(str1) < n or len(str2) < n:
    return 0
  
  # tuple of characters -> string
  str1_ngrams = list("".join(ngram) for ngram in ngrams(str1, n=n))
  str2_ngrams = list("".join(ngram) for ngram in ngrams(str2, n=n))
  
  mapping = {}
  for ngram in set(str1_ngrams + str2_ngrams):
    code = chr(65 + len(mapping))
    mapping[ngram] = code
  
  # translate
  str1_translated = "".join(mapping[ngram] for ngram in str1_ngrams)
  str2_translated = "".join(mapping[ngram] for ngram in str2_ngrams)
  return lcsr(str1_translated, str2_translated) if simdist == "sim" else normalized_edit_distance(str1_translated, str2_translated)


# Alternative implementation using unicodedata.category
def remove_diacritics_alt(text):
    """
    Alternative implementation using character categories.
    """
    normalized = unicodedata.normalize('NFD', text)
    return ''.join(char for char in normalized 
                   if unicodedata.category(char) != 'Mn')


def featurize(instance, dial):
  de_term = instance["de_term"]#.lower()
  bar_term = instance[dial if dial in instance else "dial_var"]#.lower()
  
  return [
    # PREFIX
    prefix(de_term, bar_term),
    # DICE
    dice(de_term, bar_term, n=2),
    # TRIGRAM
    dice(de_term, bar_term, n=3),
    # XDICE
    xdice(de_term, bar_term),
    # XXDICE
    xdice(de_term, bar_term, do_xxdice=True),
    # LCSR
    lcsr(de_term, bar_term),
    # NED
    normalized_edit_distance(de_term, bar_term),
    # cologne phonetics
    compute_cologne_phonetics(de_term, bar_term),
    
    # BI-SIM
    nsimdist(de_term, bar_term, n=2, simdist="sim"),
    # TRI-SIM
    nsimdist(de_term, bar_term, n=3, simdist="sim"),
    
    # BI-DIST
    nsimdist(de_term, bar_term, n=2, simdist="dist"),
    # TRI-DIST
    nsimdist(de_term, bar_term, n=3, simdist="dist"),
  ]
