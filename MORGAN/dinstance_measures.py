import numpy as np
import math
import re
from collections import Counter

WORD = re.compile(r"\w+")


def cosine_distance(str1, str2):
    vec1 = text_to_vector(str1)
    vec2 = text_to_vector(str2)
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

def levenshtein_ratio_and_distance(s, t):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                cost = 2
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions

    # Computation of the Levenshtein Distance ratio
    ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
    return ratio


if __name__ == '__main__':
    str1 = "Juri is a good guys"
    str2 = "Guys good a is Juri"
    ratio = levenshtein_ratio_and_distance(str1, str2)
    print(f"String 1 = {str1}")
    print(f"String 2 = {str2}")
    print("levenshtein distance")
    if (ratio > 0.49):
        print(f"Match with {ratio:9.4f}")
    else:
        print(f"No Match with {ratio:9.4f}")
    print("cosine distance")
    ratio = cosine_distance(str1, str2)
    if (ratio > 0.49):
        print(f"Match with {ratio:9.4f}")
    else:
        print(f"No Match with {ratio:9.4f}")