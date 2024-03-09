print("以下是题目三的输出结果：")

price_sum = 0

price_sum += 18
price_sum += 48
price_sum += 68
price_sum += 10 * 4

average = price_sum / 4

print(average)

print("以下是题目四的输出结果：")

tree1 = 1.0
tree2 = 1.0

for i in range( 1, 13, 1):
    tree1 += 0.1
    tree2 *= 1.1
    
print( "第一棵树一年后的高度为：", tree1)
print( "第二棵树一年后的高度为：", tree2)