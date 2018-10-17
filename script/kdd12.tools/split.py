#! /usr/bin/python3
import random as rd

tr = open("user.tr.ffm","w")
te = open("user.te.ffm","w")
va = open("user.va.ffm","w")

if __name__ == "__main__":
    f = open("user.ffm", "r")
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
