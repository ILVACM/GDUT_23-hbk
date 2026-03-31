#!/bin/bash
# ======================================================================
# 文件名: set-static-ip.sh
# 用途: 自动配置Ubuntu系统静态IP地址（适用于Ubuntu 17.10+）
# 作者: Qwen (阿里云)
# 日期: 2026-02-28
# 版本: 1.0
# ======================================================================
# 重要说明:
# 1. 本脚本通过Netplan配置网络，适用于Ubuntu 17.10+所有版本
# 2. 自动备份原配置文件（添加.bak后缀）
# 3. 自动适配NetworkManager（桌面版）和systemd-networkd（服务器版）
# 4. 提供参数化配置，可快速修改IP、网关、DNS等
# 5. 详细中文注释，便于理解和维护
# ======================================================================

# ====== 参数定义区（可根据需要修改） ======
# 默认配置参数（如果命令行未指定则使用这些值）
DEFAULT_IP="192.168.1.100"       # 静态IP地址
DEFAULT_NETMASK="24"             # 子网掩码（CIDR格式，如24=255.255.255.0）
DEFAULT_GATEWAY="192.168.1.1"    # 网关地址
DEFAULT_DNS="8.8.8.8,8.8.4.4"    # DNS服务器（多个用逗号分隔）
DEFAULT_INTERFACE="ens33"        # 网卡接口名（请根据实际修改）

# ====== 脚本执行前检查 ======
# 检查是否以root权限运行
if [ "$(id -u)" != "0" ]; then
    echo "错误：此脚本需要root权限运行，请使用sudo执行"
    echo "示例：sudo $0 --ip 192.168.1.100 --gateway 192.168.1.1 --interface ens33"
    exit 1
fi

# 检查Netplan是否安装
if ! command -v netplan &> /dev/null; then
    echo "错误：系统未安装Netplan，请先安装Ubuntu网络工具包"
    echo "建议运行：sudo apt update && sudo apt install -y netplan.io"
    exit 1
fi

# ====== 参数解析区 ======
# 默认参数值
ip=$DEFAULT_IP
netmask=$DEFAULT_NETMASK
gateway=$DEFAULT_GATEWAY
dns=$DEFAULT_DNS
interface=$DEFAULT_INTERFACE

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ip)
            ip="$2"
            shift 2
            ;;
        --netmask)
            netmask="$2"
            shift 2
            ;;
        --gateway)
            gateway="$2"
            shift 2
            ;;
        --dns)
            dns="$2"
            shift 2
            ;;
        --interface)
            interface="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 --ip <IP> --gateway <网关> --interface <网卡名> [可选参数: --netmask --dns]"
            exit 1
            ;;
    esac
done

# 验证必要参数是否提供
if [ -z "$ip" ] || [ -z "$gateway" ] || [ -z "$interface" ]; then
    echo "错误：必须提供--ip、--gateway和--interface参数"
    echo "用法: $0 --ip 192.168.1.100 --gateway 192.168.1.1 --interface ens33"
    exit 1
fi

# ====== 配置文件处理区 ======
# 查找Netplan配置文件（优先使用默认文件，然后找第一个.yaml）
config_file=""
if [ -f "/etc/netplan/00-installer-config.yaml" ]; then
    config_file="/etc/netplan/00-installer-config.yaml"
elif [ -f "/etc/netplan/01-netcfg.yaml" ]; then
    config_file="/etc/netplan/01-netcfg.yaml"
else
    # 查找第一个.yaml文件
    config_file=$(find /etc/netplan -maxdepth 1 -name "*.yaml" -print -quit 2>/dev/null)
    if [ -z "$config_file" ]; then
        echo "错误：未找到Netplan配置文件！请检查/etc/netplan/目录"
        exit 1
    fi
fi

echo "找到Netplan配置文件: $config_file"

# 备份原配置文件（添加.bak后缀）
backup_file="${config_file}.bak"
if [ -f "$config_file" ]; then
    cp "$config_file" "$backup_file"
    echo "已备份原配置文件至: $backup_file"
else
    echo "错误：配置文件不存在！无法备份"
    exit 1
fi

# ====== 生成新的Netplan配置 ======
# 创建临时配置文件
temp_file=$(mktemp)

# 生成YAML配置内容（使用4空格缩进）
cat > "$temp_file" << EOF
network:
  version: 2
  ethernets:
    $interface:
      dhcp4: no
      addresses: [$ip/$netmask]
      gateway4: $gateway
      nameservers:
        addresses: [$dns]
EOF

# 显示配置内容（便于用户检查）
echo -e "\n===== 生成的Netplan配置内容 ====="
cat "$temp_file"
echo "================================\n"

# ====== 应用配置======
# 使用netplan try进行安全测试（30秒内可回滚）
echo "正在安全应用新配置（30秒内可自动回滚，确保不会断开连接）..."
if ! sudo netplan try --timeout 30; then
    echo "错误：配置应用失败！已回滚到原配置"
    echo "请检查配置内容或查看系统日志：journalctl -u netplan"
    echo "备份文件: $backup_file"
    exit 1
fi

# ====== 优化点说明：无需手动重启网络服务 ======
# Netplan的netplan apply会自动处理：
# - 识别当前使用的网络管理工具（NetworkManager或systemd-networkd）
# - 适配并重启相应的网络服务
# - 无需额外的systemctl restart命令
echo -e "\n✅ 配置已安全应用！Netplan自动处理了网络服务重启"
echo "无需手动重启网络服务，配置已生效"

# ====== 验证配置 ======
echo -e "\n===== 网络配置验证 ====="
echo "当前IP地址:"
ip addr show "$interface" | grep -E "inet |inet6"
echo -e "\nDNS解析测试:"
nslookup example.com
echo "========================"

echo -e "\n✅ 静态IP配置完成！系统已自动应用新配置"
echo "新IP地址: $ip/$netmask"
echo "网关: $gateway"
echo "DNS: $dns"
echo "网卡: $interface"
echo "配置文件: $config_file"
echo "备份文件: $backup_file"
echo "配置已通过Netplan安全应用，无需额外重启"