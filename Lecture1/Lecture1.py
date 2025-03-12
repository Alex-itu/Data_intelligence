l = [1, 2, 3] 
l[0]
l.append(4)

ll = [1, "yo", {}]

l[1] = 123


t = (1, 2, 3)
t .append(1) # <- cant do
t[2] = 123 # <- cant do

d = {"key": 123, "key2": "str"}

def f(a):
    a.append(1)
    
l = [1, 2, ,3]
f(l)

# character set
"ø".encode("utf8") # <- gives one thing
"ø".encode("latin1") # <- gives another
