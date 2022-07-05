from sys import argv

def read_sequences(list_of_lines):
        """
        read a txt file and load each sequence in a dict key
        """
        seqnum = len(list_of_lines)
        n = 0
        mydict = dict()
        for line in list_of_lines:
                record = line.split()
                key = record[0]+"_"+str(n)
                val = record[1]
                mydict[key] = val
                n = n + 1
        if n != seqnum:
                print("error in reading sequences")
                raise ValueError
        return mydict,seqnum

def count_bases(seq):
        """
        check composition of given sequence
        """
        bases = {"A":0,"C":0,"G":0,"T":0}
        N = len(seq[1])
        # what about the order of the loops and the count method
        for base in bases.keys():
                bases[base] = seq[1].count(base)
        print("Composition of sequence ",seq[0]," is ")
        # not sure that everything is ok
        for i in bases.items():
                print(i[0],i[1]/N)
        
#initialization  
if __name__ == "__main__":      
	data = open(argv[1],"r").readlines()
	sequence_dict,seqnum = read_sequences(data)
	print("there are ",seqnum," sequences in the file")

	#main loop
	for item in sequence_dict.items():
		count_bases(item)

