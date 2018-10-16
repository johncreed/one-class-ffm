import sys
from random import random


ratio = 0.1
a = sys.argv[1]


c = 0
_file = open(a)
for i in _file:
    if c%10 == 0:
        print i.strip()
    c+=1
