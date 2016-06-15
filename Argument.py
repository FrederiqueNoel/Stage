import sys
import getopt
import os

def usageE():
    print("usage: ", sys.argv[0], "-m arg")
    print("0 pour gradient à pas fixe")
    print("1 pour gradient à pas optimal")
    print("2 pour gradient conjugue RR3")
    print("3 pour gradient conjugue non lineaire (Par Defaut)")

def usageC():
    print("usage: ", sys.argv[0], "-p")
    print("-p si on veut le preconditionneur")

def mainE(argv):
    methode = 3
    try:
        opts, args = getopt.getopt(argv, "hm:",["help","methode"])
    except getopt.GetoptError:
        usageE()
        sys.exit()
    
    for opt, arg in opts:
        if opt == "-h":
            usageE()
            sys.exit()
        elif opt in ("-m"):
            methode = arg

    return methode

def mainC(argv):
    pre = 0
    try:
        opts, args = getopt.getopt(argv, "hp",["help","preconditionneur"])
    except getopt.GetoptError:
        usageC()
        sys.exit()

    for opt, arg in opts:
        if opt == "-h":
            usageC()
            sys.exit()
        elif opt in ("-p"):
            pre = 1

    return pre
