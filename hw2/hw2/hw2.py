import sys
import math
import string


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)


def shred(filename):
    # Using a dictionary here. You may change this to any data structure of
    # your choice such as lists (X=[]) etc. for the assignment
    X = {char: 0 for char in string.ascii_uppercase}
    
    with open(filename, encoding='utf-8') as f:
        for line in f:
            for char in line.upper():
                if char.isalpha() and char in X:
                    X[char] += 1
    return X


def calculation(filename):
    X = shred(filename)
    e, s = get_parameter_vectors()

    def calculate_total_probabilities(letter_counts, probabilities, base_prob):
        return sum(count * math.log(prob) for count, prob in zip(letter_counts, probabilities)) + math.log(base_prob)

    print('Q2')
    print('{:.4f}'.format(X['A'] * math.log(e[0])))
    print('{:.4f}'.format(X['A'] * math.log(s[0])))

    total_English = calculate_total_probabilities(X.values(), e, 0.6)
    total_Spanish = calculate_total_probabilities(X.values(), s, 0.4)

    print('Q3')
    print('{:.4f}'.format(total_English))
    print('{:.4f}'.format(total_Spanish))

    print('Q4')
    Z = total_Spanish - total_English
    if Z >= 100:
        PYEnglish = 0
    elif Z <= -100:
        PYEnglish = 1
    else:
        PYEnglish = '{:.4f}'.format(1 / (1 + math.exp(Z)))  # using math.exp for clarity

    print(PYEnglish)


def main():
    print('Q1')
    X = shred('letter.txt')
    for key, value in X.items():
        print(key, value)

    calculation('letter.txt')


if __name__ == "__main__":
    main()