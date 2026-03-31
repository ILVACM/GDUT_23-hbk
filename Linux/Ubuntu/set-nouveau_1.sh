#!/bin/bash
# ======================================================================
# 文件名: disable-nouveau.sh
# 用途: 自动禁用Nouveau开源驱动（为安装NVIDIA显卡驱动做准备）
# 作者: Qwen (阿里云)
# 日期: 2026-02-28
# 版本: 1.0
# ======================================================================
# 重要说明:
# 1. 本脚本专为Ubuntu 17.10+系统设计
# 2. 通过编辑/etc/modprobe.d/blacklist-nouveau.conf实现禁用
# 3. 自动备份原配置文件（添加.bak后缀）
# 4. 提供参数化配置：可选择禁用(Nouveau)或启用(Nouveau)
# 5. 详细中文注释，确保脚本可维护性
# ======================================================================

# ====== 参数定义区（可按需修改） ======
# 默认操作：禁用Nouveau（安装NVIDIA驱动前必须禁用）
DEFAULT_ACTION="disable"  # 可选值: "disable" (禁用) 或 "enable" (启用)

# ====== 脚本执行前检查 ======
# 检查是否以root权限运行
if [ "$(id -u)" != "0" ]; then
    echo "错误：此脚本需要root权限运行，请使用sudo执行"
    echo "示例：sudo $0 --action disable"
    exit 1
fi

# 检查modprobe.d目录是否存在
if [ ! -d "/etc/modprobe.d" ]; then
    echo "错误：/etc/modprobe.d目录不存在！系统可能不完整"
    exit 1
fi

# ====== 参数解析区 ======
# 默认参数值
action=$DEFAULT_ACTION

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --action)
            action="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 --action <disable|enable>"
            echo "示例: sudo $0 --action disable"
            exit 1
            ;;
    esac
done

# 验证action参数是否有效
if [ "$action" != "disable" ] && [ "$action" != "enable" ]; then
    echo "错误：--action参数必须是'disable'或'enable'"
    echo "用法: $0 --action disable"
    exit 1
fi

# ====== 配置文件路径 ======
config_file="/etc/modprobe.d/blacklist-nouveau.conf"
backup_file="${config_file}.bak"

# ====== 操作执行区 ======
echo "===== Nouveau驱动配置操作开始 ====="
echo "操作类型: $action"
echo "配置文件: $config_file"
echo "备份文件: $backup_file"

# ====== 根据操作类型执行 ======
if [ "$action" = "disable" ]; then
    echo -e "\n[步骤1] 禁用Nouveau开源驱动"
    
    # 检查配置文件是否存在
    if [ -f "$config_file" ]; then
        echo "检测到原配置文件，正在备份..."
        cp "$config_file" "$backup_file"
        echo "已备份原配置文件至: $backup_file"
    else
        echo "未找到配置文件，将创建新文件"
    fi
    
    # 创建/覆盖禁用配置
    echo "blacklist nouveau" > "$config_file"
    echo "已写入禁用配置: blacklist nouveau"
    
    # 验证配置
    if grep -q "blacklist nouveau" "$config_file"; then
        echo -e "\n✅ 配置成功：Nouveau已禁用"
        echo "配置内容: $(cat "$config_file")"
    else
        echo "错误：配置写入失败，未找到禁用条目"
        exit 1
    fi

elif [ "$action" = "enable" ]; then
    echo -e "\n[步骤1] 启用Nouveau开源驱动"
    
    # 检查配置文件是否存在
    if [ -f "$config_file" ]; then
        echo "检测到配置文件，正在备份..."
        cp "$config_file" "$backup_file"
        echo "已备份原配置文件至: $backup_file"
        
        echo "正在删除禁用配置..."
        rm -f "$config_file"
        echo "已删除配置文件: $config_file"
        
        # 验证是否已删除
        if [ ! -f "$config_file" ]; then
            echo -e "\n✅ 配置成功：Nouveau已启用"
            echo "Nouveau驱动将随系统启动加载"
        else
            echo "错误：配置文件删除失败"
            exit 1
        fi
    else
        echo "未找到配置文件，Nouveau已处于启用状态"
        echo "无需操作，Nouveau驱动将随系统启动加载"
    fi
fi

# ====== 系统信息验证 ======
echo -e "\n===== 验证当前Nouveau状态 ====="
if lsmod | grep -q nouveau; then
    echo "⚠️ 警告：Nouveau驱动仍在加载中！"
    echo "请重启系统使配置生效：sudo reboot"
else
    echo "✅ 状态：Nouveau驱动已禁用（或已启用，根据操作）"
fi

echo -e "\n===== 操作完成 ====="
echo "操作类型: $action"
echo "配置文件: $config_file"
echo "备份文件: $backup_file"

if [ "$action" = "disable" ]; then
    echo -e "\n重要提示：为使配置生效，需重启系统"
    echo "建议执行：sudo reboot"
    echo "重启后，Nouveau驱动将不再加载，可安全安装NVIDIA驱动"
else
    echo -e "\n重要提示：Nouveau驱动已启用，系统重启后将使用开源驱动"
fi

echo -e "\n✅ Nouveau驱动配置操作完成！"