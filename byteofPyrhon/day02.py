def say_hello():
    # 该块属于这一函数
    print('hello world')


say_hello()


def print_max(a, b):
    if a > b:
        print(a, 'is maximum')
    elif a == b:
        print(a, 'is equal to', b)
    else:
        print(b, 'is maximum')


print_max(3, 4)
x = 5
y = 7
# 以参数的形式传递变量
print_max(x, y)

x = 50


def func(x):
    print('x is', x)
    x = 2
    print('Changed local x to', x)


func(x)
print('x is still', x)

x = 50


def func():
    # 声明 x 是一个全局变量
    global x
    print('x is', x)
    x = 2
    print('Changed global x to', x)


func()
print('Value of x is', x)


# 默认参数值
def say(message, times=1):
    print(message * times)


say('Hello')
say('World', 5)

# 关键字参数


def func(a, b=5, c=10):
    print('a is', a, 'and b is', b, 'and c is', c)


func(3, 7)
func(25, c=24)
func(c=50, a=100)


def total(a=5, *numbers, **phonebook):
    print('a', a)
    # 遍历元组中的所有项目
    for single_item in numbers:
        print('single_item', single_item)
    # 遍历字典中的所有项目
    for first_part, second_part in phonebook.items():
        print(first_part, second_part)


print(total(10, 1, 2, 3, Jack=1123, John=2231, Inge=1560))


def maximum(x, y):
    '''print the doc

    the max number of the input two numbers '''
    if x > y:
        return x
    elif x == y:
        return 'The numbers are equal'
    else:
        return y


print(maximum(2, 3))
print(maximum.__doc__)
