Input = 'An apple a day keeps doctor away!'
output = ''
word = ''
stack = []
for i in range(len(Input)):
    if Input[i]!=' ':
        word+=Input[i]
    else:
        stack.append(word)
        word = ''
stack.append(word)
print(stack)
word = stack.pop()
while  stack:
    output+=word+' '
    word = stack.pop()
output +=word
print(output)

