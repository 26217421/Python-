import sys
from math import sqrt
# 引入模块
print('\n\nThe PYTHONPATH is', sys.path, '\n')

print("Square root of 16 is", sqrt(16))

'''作用域
正常的函数和变量名是公开的（public），可以被直接引用，比如：abc，x123，PI等；
类似__xxx__这样的变量是特殊变量，可以被直接引用，但是有特殊用途，比如__author__，
__name__就是特殊变量，hello模块定义的文档注释也可以用特殊变量__doc__访问，
自己的变量一般不要用这种变量名；
类似_xxx和__xxx这样的函数或变量就是非公开的（private），不应该被直接引用，
比如_abc，__abc等；
'''
# 数据结构
'''
四种内置的数据结构——列表（List）、元组（Tuple）、字典（Dictionary）和集合（Set）
list是一种有序的集合，可以随时添加和删除其中的元素。
另一种有序列表叫元组：tuple。tuple和list非常类似，但是tuple一旦初始化就不能修改
集合（Set）是简单对象的无序集合（Collection）
'''
shoplist = ['apple', 'mango', 'carrot', 'banana']
print('I have', len(shoplist), 'items to purchase.')
print('These items are:', end=' ')
for item in shoplist:
    print(item, end=' ')
print('\nI also have to buy rice.')
shoplist.append('rice')
print('My shopping list is now', shoplist)
print('I will sort my list now')
shoplist.sort()
print('Sorted shopping list is', shoplist)
print('The first item I will buy is', shoplist[0])
olditem = shoplist[0]
del shoplist[0]
print('I bought the', olditem)
print('My shopping list is now', shoplist)
# 元组
zoo = ('python', 'elephant', 'penguin')
print('Number of animals in the zoo is', len(zoo))
new_zoo = 'monkey', 'camel', zoo
print('Number of cages in the new zoo is', len(new_zoo))
print('All animals in new zoo are', new_zoo)
print('Animals brought from old zoo are', new_zoo[2])
print('Last animal brought from old zoo is', new_zoo[2][2])
print('Number of animals in the new zoo is', len(new_zoo)-1+len(new_zoo[2]))

s = set([1, 2, 3])
print(s)
s.add(4)
s.remove(1)

d = {'Michael': 95, 'Bob': 75, 'Tracy': 85}
print(d['Michael'])
print(d.get('Thomas', -1))

# 高级特性
shoplist = ['apple', 'mango', 'carrot', 'banana']
name = 'swaroop'
# Indexing or 'Subscription' operation #
# 索引或“下标（Subscription）”操作符 #
print('Item 0 is', shoplist[0])
print('Item 1 is', shoplist[1])
print('Item 2 is', shoplist[2])
print('Item 3 is', shoplist[3])
print('Item -1 is', shoplist[-1])
print('Item -2 is', shoplist[-2])
print('Character 0 is', name[0])
# Slicing on a list #
print('Item 1 to 3 is', shoplist[1:3])
print('Item 2 to end is', shoplist[2:])
print('Item 1 to -1 is', shoplist[1:-1])
print('Item start to end is', shoplist[:])
# 从某一字符串中切片 #
print('characters 1 to 3 is', name[1:3])
print('characters 2 to end is', name[2:])
print('characters 1 to -1 is', name[1:-1])
print('characters start to end is', name[:])

print(shoplist[::-1])
