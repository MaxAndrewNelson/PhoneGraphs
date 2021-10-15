import sys

def chars_in(file):
    data = open(file, "r").read().replace('\n', ' ').split(' ') 

    unique = set(data)
    if ' ' in unique:
        unique.remove(' ')
    if '' in unique:
        unique.remove('')
        
    return list(unique)

if __name__=="__main__":
    f = sys.argv[1]

    print(' '.join(chars_in(f)))


