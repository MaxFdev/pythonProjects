import sys

st = input("choose a number.\n")
a = 0
if str.isdigit(st):
    a = int(st)
else: 
    print("Enter a digit.")
    sys.exit()
int = int(a)
string = "Did you put 1?"
print(string + " [...] " + str(int))

if int == 1:
    print("yes")
else:
    print("no")
