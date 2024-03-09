<h1 align = "center"> Git Instructions Summary </h1>

## 阅读说明

> 1.正文内容中圆弧括号'()'代表可选，即可省略  
> 2.此文针对 Git命令提示符(Git Bash) 操作环境，类似 DOS 操作方式  
> 3.一般命令格式为：$ git [cmd] (- auxiliary character) ( [argv1] …… )  
> 4.'$' 一般已经给出，'git' 基本是所有命令的前缀，此两项可能省略（有特殊情况会额外说明）  
> 5.部分指令操作需要注意操作位置和文件路径问题，即 cd 指令的结合使用
> 6.默认主分支由 master 更改为 main ，可能与政治原因有关，在指令输入时注意

## Auxiliary Characters

* -h (--)help

> 指令使用查询/帮助文档
>
> * $ git help [cmd] (argvs)

## User

* config

> * 设置查询更改（版本库，系统，全局……）指令
> *
> * $ git config --list
> *
> * $ git config (--global) user.name
> * $ git config (--global) user.email
> *
> * $ git config (--global) origin_name "new_name"  
> * $ git config (--global) origin_email "new_email"  

## WorkArea

* clone

> 下载（拷贝）项目
>
> * $ git clone [url]
>
> url : Uniform Resource Locator 统一资源定位符（网址等）

* init (initialize)

> 初始化指令
>
> * $ git init
> * $ git init [project-name]
> * $ git init --bare \<directory-name>

* clean

## Repository(self/unite)

* fetch
* pull
* push

## History Change (self)

* branch
* commit
* merge
* rebase
* reset
* switch
* tag

## File Change

* add
* mv
* rm
* restore

## Examine The history && state

* bisect
* diff
* grep
* log
* show
* status

> 分支状态查询指令
>
> * $ git status

## Reference Website

* [Git 命令参考手册整理](https://zhuanlan.zhihu.com/p/389814854)

* [Git-Github 基础操作](https://blog.csdn.net/Hanani_Jia/article/details/77950594)
