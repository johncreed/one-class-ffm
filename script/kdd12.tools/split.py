#! /usr/bin/python3
import random as rd
import sys

if len(sys.argv) != 2:
    print("./split file")
    sys.exit()

prc_filename = lambda x : (x[:x.rfind(".")] ,x[x.rfind(".")+1:])
name_tpl = prc_filename(sys.argv[1])

tr = open("{}.tr.{}".format(name_tpl[0], name_tpl[1]),"w")
te = open("{}.te.{}".format(name_tpl[0], name_tpl[1]),"w")
va = open("{}.va.{}".format(name_tpl[0], name_tpl[1]),"w")

if __name__ == "__main__":
    try:
        f = open(sys.argv[1], "r")
    except IOError:
        print("File not found.")
        sys.exit()
    rd.seed(0)
    num = 0
    for line in f:
        num = rd.randint(0, 9)
        if num == 0:
            te.write(line)
        elif num == 1:
            va.write(line)
        else:
            tr.write(line)
