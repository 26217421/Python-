# coding=UTF-8
class Person:
    def __init__(self, name):
        self.name = name

    def say_hi(self):
        print('Hello, my name is?', self.name)


p = Person('Michael')
print(p)
p.say_hi()


class Robot:
    _population = 0

    def __init__(self, name):
        self.name = name
        print("(Initializing {})".format(self.name))
        # 当有人被创建时，机器人
        # 将会增加人口数量
        Robot._population += 1

    def die(self):
        print("{} is being destroyed!".format(self.name))
        Robot._population -= 1
        if Robot._population == 0:
            print("{} was the last one.".format(self.name))
        else:
            print("There are still {:d} robots working.".format(
                Robot._population))

    def say_hi(self):
        print("Greetings, my masters call me {}.".format(self.name))

    @classmethod
    def how_many(cls):
        print("We have {:d} robots.".format(cls._population))


droid1 = Robot("R2-D2")
droid1.say_hi()
Robot.how_many()
droid2 = Robot("C-3PO")
droid2.say_hi()
Robot.how_many()
print("\nRobots can do some work here.\n")
print("Robots have finished their work. So let's destroy them.")
droid1.die()
droid2.die()
Robot.how_many()


class SchoolMember:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        print('(Initialized SchoolMember: {})'.format(self.name))

    def tell(self):
        print('Name:"{}" Age:"{}"'.format(self.name, self.age), end=" ")


class Teacher(SchoolMember):
    def __init__(self, name, age, salary):
        SchoolMember.__init__(self, name, age)
        self.salary = salary
        print('(Initialized Teacher: {})'.format(self.name))

    def tell(self):
        SchoolMember.tell(self)
        print('Salary: "{:d}"'.format(self.salary))


class Student(SchoolMember):
    def __init__(self, name, age, marks):
        SchoolMember.__init__(self, name, age)
        self.marks = marks
        print('(Initialized Student: {})'.format(self.name))

    def tell(self):
        SchoolMember.tell(self)
        print('Marks: "{:d}"'.format(self.marks))


t = Teacher('Mrs. Shrividya', 40, 30000)
s = Student('Swaroop', 25, 75)
# 打印一行空白行
print()
members = [t, s]
for member in members:
    member.tell()
