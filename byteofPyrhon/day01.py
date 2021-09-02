# 注释
print('learning started')
'''
多行注释
python 字面常量值不可变
数字主要分为两种类型——整数（Integers）与浮点数（Floats）
int类型可以指任何大小的整数
'''
age = 20
name = 'Swaroop'
print('{0} was {1} years old when he wrote this book'.format(name, age))
# 对于浮点数 '0.333' 保留小数点(.)后三位
print('{0:.3f}'.format(1.0/3))
# 使用下划线填充文本，并保持文字处于中间位置
# 使用 (^) 定义 '___hello___'字符串长度为 11
print('{0:_^11}'.format('hello'))
# 基于关键词输出 'Swaroop wrote A Byte of Python'
print('{name} wrote {book}'.format(name='Swaroop', book='A Byte of Python'))
print('a', end=' ')
print('b', end=' ')
print('c')
'''
Python 是强（Strongly）面向对象的，因为所有的一切都是对象， 包括数字、字符串与
函数。
'''

length = 5
breadth = 2
area = length * breadth
print('Area is', area)
print('Perimeter is', 2 * (length + breadth))

number = 23
guess = int(input('Enter an integer : '))

if guess == number:
    print('Congratulations, you guessed it.')
    print('(but you do not win any prizes!)')
elif guess < number:
    print('No, it is a little higher than that')
else:
    print('No, it is a little lower than that')
print('Done')

number = 23
running = True
while running:
    guess = int(input('Enter an integer : '))
    if guess == number:
        print('Congratulations, you guessed it.')
        running = False
    elif guess < number:
        print('No, it is a little higher than that.')
    else:
        print('No, it is a little lower than that.')
else:
    print('The while loop is over.')
print('Done')

for i in range(1, 5):
    print(i)
else:
    print('The for loop is over')

while True:
    s = input('Enter something : ')
    if s == 'quit':
        break
    print('Length of the string is', len(s))
print('Done')
