
"""
Finds the set of longest common substrings between two input strings with dynamic programming.
At the end we have the length of characters that match between the input strings and we output a boolean value depending 
on the similarity between the two strings.
Output:
- True -> if (100 * "threshold")% of the characters in one of the strings overlap with the characters of the other string
- False -> otherwise
"""
def relaxedStringSimilarity(s1, s2, threshold):
    
    currentLongestSubstringLength= 0
    
    # normalizing the characters
    s1= s1.lower()
    s2= s2.lower()
    
    # matrix used in the algorithm initialized with zeros
    matrix= [[0 for x in range(len(s2) - 1)] for y in range(len(s1) - 1)]
    
    
    for i in range(0, len(s1) - 1):
        for j in range(0, len(s2) - 1):
            
            currentValue= 0
            
            if s1[i] == s2[j]:
                
                if (i == 0 ) or (j == 0):
                    matrix[i][j] = 1
                    currentValue= 1
                else:
                    currentValue= matrix[i-1][j-1] + 1
                    matrix[i][j] = currentValue
            else:
                matrix[i][j]= 0
            
            if currentValue > currentLongestSubstringLength:
                currentLongestSubstringLength = currentValue
    
    
    # if (100 * "threshold")% of the characters in one of the strings overlap with the characters of the other string
    if ( ( float(currentLongestSubstringLength) / float(len(s1))) >= threshold ) and ( ( float(currentLongestSubstringLength) / float(len(s2))) >= threshold ):
        return True
    else:
        return False

"""
Finds the set of longest common substrings between two input strings with dynamic programming.
At the end we have the length of characters that match between the input strings and we output a boolean value indicating whether one of the strings can be considered a substring of the other 
Output:
- True -> if (100 * "threshold")% of the characters of the smallest input string in size matches the other input string (we can say that one of the strings is approximately a substring of the other one) 
- False -> otherwise
"""
def relaxedSubStringSimilarity(s1, s2, threshold):
    
    currentLongestSubstringLength= 0
    
    # normalizing the characters
    s1= s1.lower()
    s2= s2.lower()
    
    # matrix used in the algorithm initialized with zeros
    matrix= [[0 for x in range(len(s2) - 1)] for y in range(len(s1) - 1)]
    
    
    for i in range(0, len(s1) - 1):
        for j in range(0, len(s2) - 1):
            
            currentValue= 0
            
            if s1[i] == s2[j]:
                
                if (i == 0 ) or (j == 0):
                    matrix[i][j] = 1
                    currentValue= 1
                else:
                    currentValue= matrix[i-1][j-1] + 1
                    matrix[i][j] = currentValue
            else:
                matrix[i][j]= 0
            
            if currentValue > currentLongestSubstringLength:
                currentLongestSubstringLength = currentValue
    
    # obtain the length of the smallest input string
    smallestStringSize= 0
    if len(s1) < len(s2):
        smallestStringSize= len(s1)
    else:
        smallestStringSize= len(s2)
    
    # if more than (100 * "threshold")% of the characters of the smallest input string are equal to the other input string, then return True (indicating 
    # that the smallest string is approximately a substring of the biggest string received as input)
    if (float(currentLongestSubstringLength) / float(smallestStringSize)) >= threshold:
        return True
    else:
        return False




