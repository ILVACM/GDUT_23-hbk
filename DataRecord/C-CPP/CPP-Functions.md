<h1 = align "center"> C++ Funtions </h1>

## Foreword

> 本片总结主要是汇总 C++ 语言中的操作及函数方法
> C++ 语言原理见"C++关键总结"
> 其中标题带有 "<>" 的，要求在调用相关函数方法时需要包含对应头文件

## \<cstdio> ( \<stdio.h> )

### struct

> 该模块注意其语法格式，尤其是 C / C++ 之间的差别  

```cpp

//Declaration Structures  
struct structure_name  
{
    int example_1 = 1;
    int example_2;

    //... Write on demand of yourself
}

//Define Structures  
struct structure_name variable_name;
```

### union

## STL

### Iterator

#### Container Support

> vector                    随机访问
> deque                     随机访问
> list                      随机访问
> set / multi-set           双向
> map / multi-map           双向
> stack                     不支持
> queue / priority-queue    不支持

#### basic operate

> General —— 正向 / 双向 / 随机
>
> > 解引用                  —— *iter / iter -> member_name
> > 正序移动（步长为1）       —— iter++ ( ++iter )
> > 比较                    —— == / !=
> > 赋值 （复合赋值）         —— *iter = value ( +=, -=, *=, /=, %=)
> 
> Reverse —— 双向 / 随机
>
> > 逆序移动（步长为1）       —— iter-- ( --iter )
> 
> Random  —— 随机
>
> 随机移动
> 随机访问
> 随机解引用
> 比较
> 间距计算

### \<vecter>

### \<stack>

### \<deque>

### \<queue>

### \<set>

### \<map>

### \<list>

### \<tuple>

## \<algorithm>

## Class

……
