import os,sys

ip1 = open(sys.argv[1],'r')
list_of_phones = [i.strip() for i in ip1.readlines()]
ip1.close()

ip2 = open(sys.argv[2],'r')
for i in ip2.readlines():
    vowel = i.strip()
    if vowel in list_of_phones:
        print vowel
ip2.close()
