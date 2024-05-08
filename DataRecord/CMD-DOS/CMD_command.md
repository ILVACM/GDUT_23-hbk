# <centre>CMD命令汇总</centre>

## 参考网站资料

* Microsoft(Windows) 官方文档

> <https://learn.microsoft.com/zh-cn/windows-server/administration/windows-commands/windows-commands#command-line-shells>
  
* CMD命令格式1

> <https://blog.csdn.net/qq_44918090/article/details/126295064>
  
* CMD命令格式2

> <https://blog.csdn.net/qq_46092061/article/details/119849648>

* CMD百科汇总

> <https://baike.baidu.com/item/%E5%91%BD%E4%BB%A4%E6%8F%90%E7%A4%BA%E7%AC%A6/998728>

## 命令格式汇总

> 以下所有命令操作全部使用英文
> 输入对大小写不敏感

* 辅助符号（操作）

> /? 或者 -help
    获取命令相关参数格式
> |
    '|' 符号前的输出和符号后的输入
> />  
    重定向输出（到指定文件）并覆盖源文件
> />>
    重定向输出（到指定文件）追加到文件末尾
> <
    从file读入cmd命令
> << cmd_text
    从命令行读取输入，直到一个与text相同的行结束
> CTRL + C
    终止在运行的命令
> CLS
    清屏（命令提示符界面）
> 方向键（上、下）
    复制历史命令
> CMD            打开另一个 Windows 命令解释程序窗口。
> COLOR          设置默认控制台前景和背景颜色。

* 文件OP

> CD             显示当前目录的名称或将其更改。
> CHDIR          显示当前目录的名称或将其更改。
> MD             创建一个目录。
> MKDIR          创建一个目录。
> RD             删除目录。
> RMDIR          删除目录。
> REN            重命名文件。
> RENAME         重命名文件。
> REPLACE        替换文件。
> ASSOC          显示或修改文件扩展名关联。
> ATTRIB         显示或更改文件属性。
> CACLS          显示或修改文件的访问控制列表(ACL)。
> ICACLS         显示、修改、备份或还原文件和目录的 ACL。

* NET

> ipconfig
> ping
> netstat
> tracert 

* system

> TASKLIST       显示包括服务在内的所有当前运行的任务。
> TASKKILL       中止或停止正在运行的进程或应用程序。
> TIME           显示或设置系统时间。

* 其他

> BREAK          设置或清除扩展式 CTRL+C 检查。（Windows 中不起作用，兼容DOS）
> BCDEDIT        设置启动数据库中的属性以控制启动加载。
> CALL           从另一个批处理程序调用这一个。
> TITLE          设置 CMD.EXE 会话的窗口标题。
> \\


> \\
CHCP           显示或设置活动代码页数。
CHKDSK         检查磁盘并显示状态报告。
CHKNTFS        显示或修改启动时间磁盘检查。
COMP           比较两个或两套文件的内容。
COMPACT        显示或更改 NTFS 分区上文件的压缩。
CONVERT        将 FAT 卷转换成 NTFS。你不能转换当前驱动器。
COPY           将至少一个文件复制到另一个位置。
DATE           显示或设置日期。
DEL            删除至少一个文件。
DIR            显示一个目录中的文件和子目录。
DISKPART       显示或配置磁盘分区属性。
DOSKEY         编辑命令行、撤回 Windows 命令并创建宏。
DRIVERQUERY    显示当前设备驱动程序状态和属性。
ECHO           显示消息，或将命令回显打开或关闭。
ENDLOCAL       结束批文件中环境更改的本地化。
ERASE          删除一个或多个文件。
EXIT           退出 CMD.EXE 程序(命令解释程序)。
FC             比较两个文件或两个文件集并显示它们之间的不同。
FIND           在一个或多个文件中搜索一个文本字符串。
FINDSTR        在多个文件中搜索字符串。
FOR            为一组文件中的每个文件运行一个指定的命令。
FORMAT         格式化磁盘，以便用于 Windows。
FSUTIL         显示或配置文件系统属性。
FTYPE          显示或修改在文件扩展名关联中使用的文件类型。
GOTO           将 Windows 命令解释程序定向到批处理程序中某个带标签的行。
GPRESULT       显示计算机或用户的组策略信息。
GRAFTABL       使 Windows 在图形模式下显示扩展字符集。
IF             在批处理程序中执行有条件的处理操作。
LABEL          创建、更改或删除磁盘的卷标。
MKLINK         创建符号链接和硬链接
MODE           配置系统设备。
MORE           逐屏显示输出。
MOVE           将一个或多个文件从一个目录移动到另一个目录。
OPENFILES      显示远程用户为了文件共享而打开的文件。
PATH           为可执行文件显示或设置搜索路径。
PAUSE          暂停批处理文件的处理并显示消息。
POPD           还原通过 PUSHD 保存的当前目录的上一个值。
PRINT          打印一个文本文件。
PROMPT         更改 Windows 命令提示。
PUSHD          保存当前目录，然后对其进行更改。
RECOVER        从损坏的或有缺陷的磁盘中恢复可读信息。
REM            记录批处理文件或 CONFIG.SYS 中的注释(批注)。
ROBOCOPY       复制文件和目录树的高级实用工具
SET            显示、设置或删除 Windows 环境变量。
SETLOCAL       开始本地化批处理文件中的环境更改。
SC             显示或配置服务(后台进程)。
SCHTASKS       安排在一台计算机上运行命令和程序。
SHIFT          调整批处理文件中可替换参数的位置。
SHUTDOWN       允许通过本地或远程方式正确关闭计算机。
SORT           对输入排序。
START          启动单独的窗口以运行指定的程序或命令。
SUBST          将路径与驱动器号关联。
SYSTEMINFO     显示计算机的特定属性和配置。
TREE           以图形方式显示驱动程序或路径的目录结构。
TYPE           显示文本文件的内容。（创建？）
VER            显示 Windows 的版本。
VERIFY         告诉 Windows 是否进行验证，以确保文件正确写入磁盘。
VOL            显示磁盘卷标和序列号。
XCOPY          复制文件和目录树。
WMIC           在交互式命令 shell 中显示 WMI 信息。
> \\
notepad + 路径               打开记事本
dxdiag                      检查DirectX信息
winver                      检查Windows版本
wmimgmt.msc                 打开windows管理体系结构（WMI）
wupdmgr                     windows 更新程序
wscript                     windows脚本设置
write                       写字板
winmsd                      系统信息
wiaacmgr                    扫描仪和相机
calc                        计算器
mplayer2                    打开windows media player
mspaint                     画图板
mstsc                       远程桌面连接
mmc                         打开控制台
dxdiag                      检查Directx信息
drwtsn32                    系统医生
devmgmt.msc                 设备管理器
notepad                     记事本
ntbackup                    系统备份和还原
sndrec32                    录音机
Sndovl32                    音量控制程序
tsshutdn                    60秒倒计时关机
taskmgr                     任务管理器
explorer                    资源管理器
progman                     程序管理器
regedit.exe                 注册表
perfmon.msc                 计算机性能监测
eventvwr                    事件查看器
net user                    查看用户
whoami                      查看当前用户
net user %username% 123456  将电脑用户密码修改为123456，%%中填写用户名称