# Create a list for every name to make easier to read files
names = []
for i in range(36): # + 1 beacuse range starts at 0
    # Add a 0 in front of the number if it is less than 10
    if i < 10:
        name = 'S0' + str(i)
    else: 
        name = 'S' + str(i)
    names.append(name)
names.pop(0) # Pop the first element of the list to match the names of the files
#print(names)

names = ['S' + str(format(i, '02')) for i in range(36)] # + 1 beacuse range starts at 0
names.pop(0) # Pop the first element of the list to match the names of the files
print(names)