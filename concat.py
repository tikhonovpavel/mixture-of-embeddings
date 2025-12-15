import sys

for f in sys.argv[1:]:
    c = '#' if f.endswith('.py') else '//'
    print(f"------------------------------------\n{c} {f}\n\n{open(f).read()}\n")