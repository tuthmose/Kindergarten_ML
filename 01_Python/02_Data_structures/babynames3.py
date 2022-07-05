
import pickle
import re
import sys

def extract_names(filename):

    f = open(filename, 'r')
    text = f.read()
    names = list()

    try:
        year_match = re.search(r'Popularity\sin\s(\d+)<', text)
        year = year_match.group(1)
        print(year)
    except:
        print("No year found")
        quit()

    lines = text.split("\n")
    #check out what you are looking at
    #print(lines[:10])
    #<tr align="right"><td>6</td><td>Daniel</td><td>Abigail</td>
    pattern = r'<td>(\d+)</td><td>(\w+)</td>\<td>(\w+)</td>'
    for line in lines:
        try:
            name_search = re.search(pattern,line)
            rank = name_search.group(1)
            male = name_search.group(2)
            female = name_search.group(3)
            names.append((male,rank))
            names.append((female,rank))
        except:
            continue
    names = sorted(names)
    #print(names[:10])
    return year, names

def load_data():
    #complete it
    # an empty function by default returns none
    return None


def main():
    args = sys.argv[1:]

    if not args:
        print('No input data')
        quit()

    data = dict()
    for filename in args:
        year,names = extract_names(filename)
        data[year] = names

    outf = open('namedb', 'wb')
    pickle.dump(data,outf)

if __name__ == '__main__':
  main()
