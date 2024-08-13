<h1 align = "center"> Java 语言格式整理 </h1>

## Foreword

该文档是用于整理学习 Java 语言中的关键点。学习 Java 时，是在已经学习了 C / C++ 和 Python 的基础上进行的。

## Java Install

### Windows

### Linux - Ubuntu

## Java VS C/C++

1. Java 是强类型语言，C / C++ 是弱类型语言，Java 的类型检查远比 C / C++ 严格
2. Java 中的变量更多的创建为引用
3. Java 中的函数参数传递（ 实参 / 形参 ）均为值传递（拷贝值，原地址值不改变）

## Reserved Word

byte short int long float double char boolean void null  
public static final  
……  

## Data Type

byte short int long  

float double  

char  

boolean  

public private static final  

## variant

### local variable

局部变量

### Instance Variables

实例变量

## Function Format

### Main Function

在 Java 中，在程序的真正的类函数入口加入（且仅使用一次）如下函数：

    ```java
        public static void main(String[] args) {}
    ```

### Encapsulation

#### Getter && Setter

Setter && Getter 并不是一个具体的函数或者方法，是 Java 对象中的一个封装处理思想（类似算法），是一种保证程序正常运行的处理流程步骤。

setter （传入）将实参赋值给对象的成员变量
getter （传出）返回对象的成员变量的值

#### Format

\[Modifier] [Type] [Name] ( [Parameter] ) {}  

#### Access Modifier

public private protected

static（静态） final（常量）

## Comment

// 单行注释

/\* …… */ 多行注释

## Operator

1. comparison operator: == != > < >= <=  
2. Relational Operator: + - * / % ++ -- 组合运算符
3. Logical Operator: && || !

