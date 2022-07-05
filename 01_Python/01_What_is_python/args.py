#!/usr/bin/env python3

from sys import argv

def dumb_function():
    for num,arg in enumerate(argv):
        print(num, arg)

dumb_function()

