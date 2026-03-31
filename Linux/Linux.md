# Linux

## Linux 系统安装分区方案

### 硬件配置实例
- 硬盘容量：512GB（实际可用约 476GiB，因 1GB=10⁹B 而 1GiB=2³⁰B）
- 运行内存：32GB

### 分区前注意事项
1. **单位换算**：硬盘厂商使用十进制（GB），Linux 使用二进制（GiB）
   - 512GB ≈ 476GiB
   - 预留 10-20GB 未分配空间作为缓冲
2. **备份数据**：分区前务必备份重要数据
3. **引导方式确认**：BIOS 和 UEFI 分区方案不同，**尤其是引导分区**

---

### 核心差异：BIOS vs UEFI 引导分区对比

| 对比项 | BIOS 引导 | UEFI 引导 |
|--------|-----------|-----------|
| **分区表类型** | MBR | GPT |
| **引导分区挂载点** | `/boot` | `/boot/efi`（EFI 系统分区） |
| **引导分区文件系统** | ext4/ext3 | **FAT32（必须）** |
| **引导分区大小** | 500MB-1GB | 512MB-1GB |
| **引导分区位置** | 必须在磁盘前 2GB | 无特殊要求 |
| **分区类型** | 主分区 | 带 ESP 标志的主分区 |
| **其他分区** | 逻辑分区（通过扩展分区） | 全部为主分区 |
| **最大分区数** | 4 个主分区（或 3 主 + 多逻辑） | 128 个主分区 |
| **最大支持磁盘** | 2TB | 支持>2TB |

> ⚠️ **关键区别**：UEFI **必须**创建 FAT32 格式的 EFI 系统分区（ESP），而 BIOS 使用普通 ext4 格式的/boot 分区

---

### BIOS 引导方式分区方案（MBR）

**分区表结构**：MBR（主引导记录）

| 分区 | 挂载点 | 文件系统 | 大小 | 分区类型 | 作用说明 |
|------|--------|----------|------|----------|----------|
| /dev/sda1 | `/boot` | ext4 | 1GB | **主分区** | **引导分区**：存放内核、initrd、GRUB 配置文件 |
| /dev/sda2 | - | - | 剩余空间 | **扩展分区** | 包含所有逻辑分区 |
| /dev/sda5 | `/` | ext4 | 100GB | 逻辑分区 | 根分区：系统和应用程序 |
| /dev/sda6 | `/home` | ext4 | 350GB | 逻辑分区 | 用户数据分区 |
| /dev/sda7 | `swap` | swap | 32GB | 逻辑分区 | 交换分区 |
| /dev/sda8 | `/var` | ext4 | 20GB | 逻辑分区 | 日志、缓存等可变数据 |

**BIOS 分区特点**：
- ✅ 使用 MBR 分区表，兼容性最好
- ⚠️ 主分区最多 4 个，需通过扩展分区创建更多逻辑分区
- ⚠️ `/boot` 分区必须位于磁盘前 2GB 范围内
- ⚠️ 不支持大于 2TB 的单个分区
- 🔧 适用于老旧硬件或需要兼容旧系统的场景

---

### UEFI 引导方式分区方案（GPT）

**分区表结构**：GPT（GUID 分区表）

| 分区 | 挂载点 | 文件系统 | 大小 | 分区标志 | 作用说明 |
|------|--------|----------|------|----------|----------|
| /dev/sda1 | `/boot/efi` | **FAT32** | 512MB-1GB | `boot`, `esp` | **EFI 系统分区（ESP）**：UEFI 引导文件、.efi 文件 |
| /dev/sda2 | `/boot` | ext4 | 1GB | - | 可选：存放内核和 initramfs |
| /dev/sda3 | `/` | ext4 | 100GB | - | 根分区：系统和应用程序 |
| /dev/sda4 | `/home` | ext4 | 350GB | - | 用户数据分区 |
| /dev/sda5 | `swap` | swap | 32GB | - | 交换分区 |
| /dev/sda6 | `/var` | ext4 | 20GB | - | 日志、缓存等可变数据 |

**UEFI 分区特点**：
- ✅ 使用 GPT 分区表，支持大于 2TB 的分区
- ✅ **必须有 EFI 系统分区（ESP）**，格式为 FAT32
- ✅ 支持更多主分区（最多 128 个），无扩展分区概念
- ✅ 引导更安全，支持 Secure Boot
- ✅ 现代计算机推荐使用 UEFI 方式
- ⚠️ EFI 分区不能格式化为 ext4 等 Linux 文件系统

---

### 引导分区详细说明

#### BIOS 引导分区（/boot）
```
位置：磁盘起始位置（前 2GB 内）
文件系统：ext4/ext3
内容：
  - vmlinuz-*      # Linux 内核文件
  - initrd.img-*   # 初始化内存盘
  - grub/          # GRUB 引导加载程序配置
  - config-*       # 内核配置文件
  - System.map-*   # 内核符号表
```

#### UEFI 引导分区（/boot/efi）
```
位置：无特殊要求（通常放在磁盘起始位置）
文件系统：FAT32（必须）
ESP 标志：必须设置
内容：
  - EFI/
    - boot/
      - bootx64.efi    # 默认引导程序
    - ubuntu/
      - grubx64.efi    # GRUB 引导程序
      - shimx64.efi    # Secure Boot 支持
  ```

---

### 各分区大小建议说明

| 分区 | 挂载点 | 建议大小 | 必要性 | 说明 |
|------|--------|----------|--------|------|
| **引导分区** | `/boot` 或 `/boot/efi` | 512MB-1GB | **必需** | BIOS 用 ext4，UEFI 用 FAT32 |
| **根分区** | `/` | 80-150GB | **必需** | 存放系统和应用程序 |
| **家目录** | `/home` | 剩余空间 70-80% | 推荐 | 独立分区便于重装系统保留数据 |
| **交换分区** | `swap` | 16-32GB | 推荐 | 32GB 内存建议配置 16-32GB |
| **变量分区** | `/var` | 15-30GB | 可选 | 服务器建议独立分区 |

**Swap 大小参考**：
- 内存≤4GB：swap = 2×内存
- 4GB<内存≤16GB：swap = 1×内存
- 内存>16GB：swap = 0.5-1×内存（或根据休眠需求，休眠需 swap≥内存）

---

### 文件系统选择建议

| 文件系统 | 适用分区 | 特点 |
|----------|----------|------|
| **FAT32** | **UEFI 的 EFI 分区** | **UEFI 引导必需**，仅用于 ESP |
| ext4 | /、/home、/boot、/var | 通用推荐，稳定成熟 |
| xfs | /home、/var | 适合大文件、高性能场景 |
| btrfs | /、/home | 支持快照、压缩、RAID |

## Ubuntu

### 安装 NVIDIA GPU drivers

1. 更新系统
   参考命令: `sudo apt update && sudo apt upgrade -y`
2. 禁用开源驱动 Nouveau
   - 编辑文件`blacklist.conf`  
    参考方式  
    > `sudo nano /etc/modprobe.d/blacklist.conf`  
    >
    > 在文件中添加内容：  
    >
    > ```shell
    > blacklist nouveau
    > options nouveau modeset=0
    > ```
    >
    > 保存并退出文件  
    >
    > 应用设置并重启系统  
    > `sudo update-initramfs -u`  
    > `sudo reboot`

3. 清理原有残留  
    参考命令:  
    1. `sudo apt autoremove --purge nvidia-*`
    2. `sudo apt purge nvidia-*`
4. 安装 NVIDIA GPU drivers
   - 系统GUI 官方仓库
   - 系统CLI 配置  
    参考方式  
    > 添加官方 PPA 仓库并更新（不一定，这一步按需操作）  
    > `sudo add-apt-repository ppa:graphics-drivers/ppa`  
    > `sudo apt update`  
    >
    > 查看可用驱动版本（关注`recommended`标识的版本）  
    > `ubuntu-drivers devices`  
    >
    > 安装指定驱动版本
    > `sudo apt install nvidia-driver-<版本号>`
    >
    > 重启系统
    > `sudo reboot`

   - 手动安装
5. 验证  
   参考命令: `nvidia-smi`


### 文件传输

1. rsync
2. WinSCP
3. rclone

| 原理         | 适用场景       | 配置难度 | 安全性  | 跨平台兼容性      |
|------------|------------|------|------|-------------|
| SCP/SFTP   | 快速传输、脚本自动化 | ⭐⭐   | 🔒 高 | ✅ 全平台       |
| Samba 共享    | 频繁互访、多人协作  | ⭐⭐⭐  | 🔒 中 | ✅ 全平台       |
| FTP/SFTP服务 | 大文件传输、图形界面 | ⭐⭐⭐  | 🔒 中 | ✅ 全平台       |
| SSHFS 挂载    | 像本地文件夹一样访问 | ⭐⭐   | 🔒 高 | ✅ Linux/Mac |
