# The 1st method

"""

第一种方法首先采用类似 C/C++ 的高精度( 详情见 OI WIKI 网站说明 )
但是相较 C/C++ ，对 python 的各类模块和语法格式极其不熟练，因此一直无法解决最后一位的精度丢失问题
但是可以做到任意位数的数字反向

"""

##################################################################
origin = input("Please input a float number(this is the first test):")

length = len(origin)-1

origin = float(origin)

inter = int(origin)

origin -= inter

num_int = []
num_float = []

i = 0
while inter:
    num_int.append( inter % 10 )
    inter = int( inter / 10 )
    i += 1
    
for j in range( 1, length-i+1, 1):
    inter = int( origin * (10**j) )
    num_float.append( inter % 10 )
    
ans = 0.0

i = 0
for j in num_float:
    ans += j * ( 10**i )
    i += 1

i = -1
for j in num_int:
    ans += j * ( 10**i )
    i -= 1
    
print(ans)




# The 2nd method

"""

第二种方法通过尽可能使用整数取数并重排列，避免了精度丢失问题
同样可以做到任意位数的数字反向

"""

######################################################

origin = input("Please input a float number(this is the second test):")

length = len(origin)-1

origin = float(origin)

inter = int(origin)

origin -= inter

num_int = []
num_float = []

i = 0
while inter:
    num_int.append( inter % 10 )
    inter = int( inter / 10 )
    i += 1
num_int.reverse()
    
decimal = length-i
    
for j in range( 1, decimal+1, 1):
    inter = int( origin * (10**j) )
    num_float.append( inter % 10 )
    
ans = 0

i = 0
for j in num_int:
    ans += j * ( 10**i )
    i += 1
for j in num_float:
    ans += j * ( 10**i )
    i += 1
    
ans /= ( 10**decimal )
    
print(ans)