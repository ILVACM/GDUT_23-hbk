<h1 align = "center"> Markdown 语言格式整理 </h1>

## 标题

1.'=','-' 标记格式

> * 语法说明：
>
> > * 可标记 2 级标题
> > * 在标题下方打等长符号
>
> * 定义格式：
>
> > 1.'=' 样式  
> >
> > First_level title_name  
> > \=====================  
> >
> > 2.'-' 样式
> >
> > First_level title_name  
> > \---------------------
>

2.'#' 标记格式

> * 语法说明：
>
> > * 可标记 6 级标题
> > * 在标题左边输入相应数量符号
>
> * 定义格式：
>
> > * \# First_level title_name
> > * \## Second_level title_name  
> > ...
> > * \###### Sixth_level title_name
>

3.标题居中

> \<h1 align = "center"> Title_context \</h1>  
> PS：其中 h1 指代一级标题，h2 指二级标题，以此类推

## 段落

> 1. 整体布局
>
> > * 段落内容：直接编写
> > * 换行：两个及以上空格 + Enter
>
> 2. 字体
>
> > * 斜体文本
> >   * \* Italic Text *
> >   * \_ Italic Text _
> > * 粗体文本
> >   * \** Bold Text **
> >   * \__ Bold Text __
> > * 粗斜体文本
> >   * \*** Bold & Italic  Text ***
> >   * \___ Bold & Italic  Text ___
>
> 3. 分割线
>
> > * 格式说明：
> >   * 一行中用三个以上的星号、减号、底线来建立一个分隔线
> >   * 行内不能有其他内容
> >   * 可以在星号或是减号中间插入空格
> > * 定义格式：
> >   * \***
> >   * \* * *
> >   * \---
> >   * \- - - -
> >   * \___
>
> 4. 删除线
>
> > * 定义格式：
> > \~~ Deleted Text ~~
>
> 5. 下划线
>
> > * 定义格式：
> > \<u> Underlined Text </u>
>
> 6. 脚注
>
> > * 定义格式：  
> > \[^Annotate-Text]: "Tnterpretative Statement or Explanation"
>
>

## 列表

> 1. 无序列表
>
> > * 定义格式：
> >   * \* Content Text
> >   * \+ Content Text
> >   * \- Content Text
>
> 2. 有序列表
>
> > * 定义格式：  
> > Numerical_code. Content Text
>
> 3. 列表嵌套
>
> > * 格式说明：  
> > 在子列表中的选项前面添加两个或四个空格 (x) 即可
> > * 定义格式：  
> > Numerical-code. Content Text  
> > xx\* Content Text  
> > xx\* Content Text
>

## 区块

## 代码

> 1. 函数/代码片段 （反引号）
>
> > \`Code Snippets`  
>
> 2. 代码块
>
> > \t Code Block  
>
> 3. 指定语言代码块 （反引号）
>
> > \```ProgramLanguage (It can be ignored)  
> > Code Block  
> > \```  

## 链接

> 1. 名称链接   \[Interlinkage - Name](Interlinkage - address)
> 2. 直接链接   \<Interlinkage - address>
> 3. 变量链接   \[Interlinkage - Name][Interlinkage - var] ... [Interlinkage - var]:Interlinkage - address

## 图片

> \!\[Attribute-Text](Image-address)

## 表格

## 高级技巧
