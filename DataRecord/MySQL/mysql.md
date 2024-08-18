<h1 align = "center"> MySQL </h1>

## Foreword

### Windows10

Basic Parameters
MySQL Corporation :         Oracle -> <https://www.oracle.com/>  
MySQL Offical Website :     <https://dev.mysql.com/>  
MySQL DataBase Version :    8.0.37  

### Linux

### Reference

[CSDN](https://blog.csdn.net/)
[CSDN-MySQL](https://edu.csdn.net/skill/mysql/)

## Installation

[MySQL 8.0 Installation Guide](https://blog.csdn.net/m0_52559040/article/details/121843945)
[MySQL 5.7 Installation Guide](https://edu.csdn.net/skill/mysql/mysql-95d0b7e5493e478f85ca49a77a13d194)

## Uninstallation

### Windows

1. Stop the MySQL service  

（命令行界面）          输入命令`net stop mysql`  

（控制面板 -> 服务）    找到 "MySQL" 相关服务，右键"停止服务"  

2. Remove the MySQL installation  

（控制面板 -> 程序）    找到 "MySQL" 相关程序，右键"卸载"  

3. Remove the MySQL data files  

（数据库文件所在位置）   删除文件夹  
默认安装路径 -> C:\ProgramData\MySQL\MySQL Server 8.0  
我实际安装路径 -> D:\MySQL  

4. Remove the MySQL configuration files

（注册表编辑器）        win键 + R -> regedit  

如下路径搜索并删除：  
HKEY_LOCAL_MACHINE/SYSTEM/ControlSet001/Services/Eventlog/Application/MySQL  
HKEY_LOCAL_MACHINE/SYSTEM/ControlSet002/Services/Eventlog/Application/MySQL  
HKEY_LOCAL_MACHINE/SYSTEM/CurrentControlSet/Services/Eventlog/Application/MySQL  

## Usage
