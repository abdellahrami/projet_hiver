file = open("rgb_3classes.txt",'r')
lines = [line.replace('\n','') for line in file]


lines_out = []
lines_out.append(lines[0])

for i in range(1,len(lines)):
    a = lines[i].split('\t')
    a = [ l.split(',') for l in a if l!='']
    lines_out.append('\t'.join(['1'+','+l[1]+','+'3' for l in a]))

file_out = '\n'.join(lines_out)
f = open('r_3classes.txt', 'w')
f.write(file_out)

f.close()
file.close()
