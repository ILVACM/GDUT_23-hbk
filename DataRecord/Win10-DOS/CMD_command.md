# <centre>CMD�������</centre>

## �ο���վ����

* Microsoft(Windows) �ٷ��ĵ�

> <https://learn.microsoft.com/zh-cn/windows-server/administration/windows-commands/windows-commands#command-line-shells>
  
* CMD�����ʽ1

> <https://blog.csdn.net/qq_44918090/article/details/126295064>
  
* CMD�����ʽ2

> <https://blog.csdn.net/qq_46092061/article/details/119849648>

* CMD�ٿƻ���

> <https://baike.baidu.com/item/%E5%91%BD%E4%BB%A4%E6%8F%90%E7%A4%BA%E7%AC%A6/998728>

## �����ʽ����

> ���������������ȫ��ʹ��Ӣ��
> ����Դ�Сд������

* �������ţ�������

> /? ���� -help
    ��ȡ������ز�����ʽ
> |
    '|' ����ǰ������ͷ��ź������
> />  
    �ض����������ָ���ļ���������Դ�ļ�
> />>
    �ض����������ָ���ļ���׷�ӵ��ļ�ĩβ
> <
    ��file����cmd����
> << cmd_text
    �������ж�ȡ���룬ֱ��һ����text��ͬ���н���
> CTRL + C
    ��ֹ�����е�����
> CLS
    ������������ʾ�����棩
> ��������ϡ��£�
    ������ʷ����
> CMD            ����һ�� Windows ������ͳ��򴰿ڡ�
> COLOR          ����Ĭ�Ͽ���̨ǰ���ͱ�����ɫ��

* �ļ�OP

> CD             ��ʾ��ǰĿ¼�����ƻ�����ġ�
> CHDIR          ��ʾ��ǰĿ¼�����ƻ�����ġ�
> MD             ����һ��Ŀ¼��
> MKDIR          ����һ��Ŀ¼��
> RD             ɾ��Ŀ¼��
> RMDIR          ɾ��Ŀ¼��
> REN            �������ļ���
> RENAME         �������ļ���
> REPLACE        �滻�ļ���
> ASSOC          ��ʾ���޸��ļ���չ��������
> ATTRIB         ��ʾ������ļ����ԡ�
> CACLS          ��ʾ���޸��ļ��ķ��ʿ����б�(ACL)��
> ICACLS         ��ʾ���޸ġ����ݻ�ԭ�ļ���Ŀ¼�� ACL��

* NET

> ipconfig
> ping
> netstat
> tracert 

* system

> TASKLIST       ��ʾ�����������ڵ����е�ǰ���е�����
> TASKKILL       ��ֹ��ֹͣ�������еĽ��̻�Ӧ�ó���
> TIME           ��ʾ������ϵͳʱ�䡣

* ����

> BREAK          ���û������չʽ CTRL+C ��顣��Windows �в������ã�����DOS��
> BCDEDIT        �����������ݿ��е������Կ����������ء�
> CALL           ����һ����������������һ����
> TITLE          ���� CMD.EXE �Ự�Ĵ��ڱ��⡣
> \\


> \\
CHCP           ��ʾ�����û����ҳ����
CHKDSK         �����̲���ʾ״̬���档
CHKNTFS        ��ʾ���޸�����ʱ����̼�顣
COMP           �Ƚ������������ļ������ݡ�
COMPACT        ��ʾ����� NTFS �������ļ���ѹ����
CONVERT        �� FAT ��ת���� NTFS���㲻��ת����ǰ��������
COPY           ������һ���ļ����Ƶ���һ��λ�á�
DATE           ��ʾ���������ڡ�
DEL            ɾ������һ���ļ���
DIR            ��ʾһ��Ŀ¼�е��ļ�����Ŀ¼��
DISKPART       ��ʾ�����ô��̷������ԡ�
DOSKEY         �༭�����С����� Windows ��������ꡣ
DRIVERQUERY    ��ʾ��ǰ�豸��������״̬�����ԡ�
ECHO           ��ʾ��Ϣ����������Դ򿪻�رա�
ENDLOCAL       �������ļ��л������ĵı��ػ���
ERASE          ɾ��һ�������ļ���
EXIT           �˳� CMD.EXE ����(������ͳ���)��
FC             �Ƚ������ļ��������ļ�������ʾ����֮��Ĳ�ͬ��
FIND           ��һ�������ļ�������һ���ı��ַ�����
FINDSTR        �ڶ���ļ��������ַ�����
FOR            Ϊһ���ļ��е�ÿ���ļ�����һ��ָ�������
FORMAT         ��ʽ�����̣��Ա����� Windows��
FSUTIL         ��ʾ�������ļ�ϵͳ���ԡ�
FTYPE          ��ʾ���޸����ļ���չ��������ʹ�õ��ļ����͡�
GOTO           �� Windows ������ͳ����������������ĳ������ǩ���С�
GPRESULT       ��ʾ��������û����������Ϣ��
GRAFTABL       ʹ Windows ��ͼ��ģʽ����ʾ��չ�ַ�����
IF             �������������ִ���������Ĵ��������
LABEL          ���������Ļ�ɾ�����̵ľ�ꡣ
MKLINK         �����������Ӻ�Ӳ����
MODE           ����ϵͳ�豸��
MORE           ������ʾ�����
MOVE           ��һ�������ļ���һ��Ŀ¼�ƶ�����һ��Ŀ¼��
OPENFILES      ��ʾԶ���û�Ϊ���ļ�������򿪵��ļ���
PATH           Ϊ��ִ���ļ���ʾ����������·����
PAUSE          ��ͣ�������ļ��Ĵ�����ʾ��Ϣ��
POPD           ��ԭͨ�� PUSHD ����ĵ�ǰĿ¼����һ��ֵ��
PRINT          ��ӡһ���ı��ļ���
PROMPT         ���� Windows ������ʾ��
PUSHD          ���浱ǰĿ¼��Ȼ�������и��ġ�
RECOVER        ���𻵵Ļ���ȱ�ݵĴ����лָ��ɶ���Ϣ��
REM            ��¼�������ļ��� CONFIG.SYS �е�ע��(��ע)��
ROBOCOPY       �����ļ���Ŀ¼���ĸ߼�ʵ�ù���
SET            ��ʾ�����û�ɾ�� Windows ����������
SETLOCAL       ��ʼ���ػ��������ļ��еĻ������ġ�
SC             ��ʾ�����÷���(��̨����)��
SCHTASKS       ������һ̨���������������ͳ���
SHIFT          �����������ļ��п��滻������λ�á�
SHUTDOWN       ����ͨ�����ػ�Զ�̷�ʽ��ȷ�رռ������
SORT           ����������
START          ���������Ĵ���������ָ���ĳ�������
SUBST          ��·�����������Ź�����
SYSTEMINFO     ��ʾ��������ض����Ժ����á�
TREE           ��ͼ�η�ʽ��ʾ���������·����Ŀ¼�ṹ��
TYPE           ��ʾ�ı��ļ������ݡ�����������
VER            ��ʾ Windows �İ汾��
VERIFY         ���� Windows �Ƿ������֤����ȷ���ļ���ȷд����̡�
VOL            ��ʾ���̾������кš�
XCOPY          �����ļ���Ŀ¼����
WMIC           �ڽ���ʽ���� shell ����ʾ WMI ��Ϣ��
> \\
notepad + ·��               �򿪼��±�
dxdiag                      ���DirectX��Ϣ
winver                      ���Windows�汾
wmimgmt.msc                 ��windows������ϵ�ṹ��WMI��
wupdmgr                     windows ���³���
wscript                     windows�ű�����
write                       д�ְ�
winmsd                      ϵͳ��Ϣ
wiaacmgr                    ɨ���Ǻ����
calc                        ������
mplayer2                    ��windows media player
mspaint                     ��ͼ��
mstsc                       Զ����������
mmc                         �򿪿���̨
dxdiag                      ���Directx��Ϣ
drwtsn32                    ϵͳҽ��
devmgmt.msc                 �豸������
notepad                     ���±�
ntbackup                    ϵͳ���ݺͻ�ԭ
sndrec32                    ¼����
Sndovl32                    �������Ƴ���
tsshutdn                    60�뵹��ʱ�ػ�
taskmgr                     ���������
explorer                    ��Դ������
progman                     ���������
regedit.exe                 ע���
perfmon.msc                 ��������ܼ��
eventvwr                    �¼��鿴��
net user                    �鿴�û�
whoami                      �鿴��ǰ�û�
net user %username% 123456  �������û������޸�Ϊ123456��%%����д�û�����