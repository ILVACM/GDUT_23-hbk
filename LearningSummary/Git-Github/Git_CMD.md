<h1 align = "center"> Git Instructions Summary </h1>

## �Ķ�˵��

> 1.����������Բ������'()'�����ѡ������ʡ��  
> 2.������� Git������ʾ��(Git Bash) �������������� DOS ������ʽ  
> 3.һ�������ʽΪ��$ git [cmd] (- auxiliary character) ( [argv1] ���� )  
> 4.'$' һ���Ѿ�������'git' ���������������ǰ׺�����������ʡ�ԣ���������������˵����  
> 5.����ָ�������Ҫע�����λ�ú��ļ�·�����⣬�� cd ָ��Ľ��ʹ��
> 6.Ĭ������֧�� master ����Ϊ main ������������ԭ���йأ���ָ������ʱע��

## Auxiliary Characters

* -h (--)help

> ָ��ʹ�ò�ѯ/�����ĵ�
>
> * $ git help [cmd] (argvs)

## User

* config

> * ���ò�ѯ���ģ��汾�⣬ϵͳ��ȫ�֡�����ָ��
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

> ���أ���������Ŀ
>
> * $ git clone [url]
>
> url : Uniform Resource Locator ͳһ��Դ��λ������ַ�ȣ�

* init (initialize)

> ��ʼ��ָ��
>
> * $ git init
> * $ git init [project-name]
> * $ git init --bare \<directory-name>

* clean

## Repository(self/unite)

* fetch
* pull
* push
* remote

> Զ��ͬ��ָ��
>
> * $ git remote -v
> * $ git remote show [remote repository name]
>
> * $ git remote add [repository_name] [URL]
> * $ git remote add
>
> * $ git remote rm <repository_name>

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

> �ļ����ָ�� ���ļ�������������У�ʹ Git ����׷�ټ�¼�ļ����޸Ĳ��ύ����
>
> * $ git add .
> * $ git add [(Drive:)/sub-directory/.../]
> *
> * $ git add specify_file.js
> * $ git add (Drive:)/sub-directory/.../specify_file.js
> * $ git add [file1] [file2] ...
> *
> * $ git add -p ���ÿ���仯ǰ������Ҫ��ȷ��

* rm

> �ļ�ɾ��ָ�� ���ļ��ӹ�������ɾ����������Ӱ����� Git �����ύ����
>
> * $ git rm specify_file.js
> * $ git rm (Drive:)/sub-directory/.../specify_file.js
> * $ git rm [file1] [file2] ...
>
> * $ git rm --cached [file] ֹͣ׷��ָ���ļ��������ļ��ᱣ���ڹ�����

* mv
* restore

## Examine The history && state

* bisect
* diff
* grep
* log
* show
* status

> ��֧״̬��ѯָ��
>
> * $ git status

## Reference Website

* [Git ����ο��ֲ�����](https://zhuanlan.zhihu.com/p/389814854)

* [Git-Github ��������](https://blog.csdn.net/Hanani_Jia/article/details/77950594)

* [Git ����ʹ���ĵ������أ�](C:\Program Files\Git\mingw64\share\doc\git-doc)
