#!/bin/bash
# ======================================================================
# 文件名: disable-nouveau.sh
# 用途: 自动禁用Nouveau开源显卡驱动（为安装NVIDIA驱动做准备）
# 作者: Qwen (阿里云)
# 日期: 2026-03-01
# 版本: 1.0
# ======================================================================
# 重要说明:
# 1. 本脚本通过修改/etc/modprobe.d/blacklist.conf实现Nouveau禁用
# 2. 自动备份原配置文件（添加.bak后缀）
# 3. 提供参数化配置：默认禁用Nouveau，可使用--enable启用
# 4. 适用于Ubuntu 17.10+所有版本（包括桌面版和服务器版）
# 5. 包含详细中文注释和安全验证
# ======================================================================

# ====== 参数定义区 ======
# 默认行为：禁用Nouveau（安装NVIDIA驱动前必须禁用）
# 如果需要启用Nouveau，请使用--enable参数
enable_nouveau=false

# ====== 脚本执行前检查 ======
# 检查是否以root权限运行
if [ "$(id -u)" != "0" ]; then
    echo "错误：此脚本需要root权限运行，请使用sudo执行"
    echo "用法: sudo $0 [--enable]"
    exit 1
fi

# 检查blacklist.conf文件是否存在
if [ ! -f "/etc/modprobe.d/blacklist.conf" ]; then
    echo "错误：系统未找到blacklist.conf文件！请确认系统配置"
    echo "Nouveau禁用需要修改/etc/modprobe.d/blacklist.conf"
    exit 1
fi

# ====== 参数解析区 ======
while [[ $# -gt 0 ]]; do
    case "$1" in
        --enable)
            enable_nouveau=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--enable]"
            echo "  --enable: 启用Nouveau驱动（默认禁用）"
            exit 1
            ;;
    esac
done

# ====== 配置文件处理区 ======
# 备份原配置文件（添加.bak后缀）
backup_file="/etc/modprobe.d/blacklist.conf.bak"
if [ -f "/etc/modprobe.d/blacklist.conf" ]; then
    cp "/etc/modprobe.d/blacklist.conf" "$backup_file"
    echo "✅ 已备份原配置文件至: $backup_file"
else
    echo "错误：配置文件不存在！无法备份"
    exit 1
fi

# ====== 核心操作：禁用或启用Nouveau ======
if [ "$enable_nouveau" = true ]; then
    echo "🔍 正在启用Nouveau驱动（移除黑名单配置）..."

    # 移除所有关于Nouveau的黑名单行（精确匹配）
    sed -i '/blacklist nouveau/d' /etc/modprobe.d/blacklist.conf

    # 验证是否移除成功
    if ! grep -q "blacklist nouveau" /etc/modprobe.d/blacklist.conf; then
        echo "✅ Nouveau驱动已启用！"
        echo "注意：启用后需重启系统才能生效"
    else
        echo "⚠️ 警告：Nouveau黑名单未完全移除，请检查配置文件"
        exit 1
    fi
else
    echo "🔍 正在禁用Nouveau驱动（添加黑名单配置）..."

    # 检查是否已禁用
    if grep -q "blacklist nouveau" /etc/modprobe.d/blacklist.conf; then
        echo "ℹ️ Nouveau驱动已禁用，无需修改"
    else
        # 添加禁用配置（确保只添加一次）
        echo "blacklist nouveau" >> /etc/modprobe.d/blacklist.conf
        echo "✅ Nouveau驱动已禁用！"
        echo "注意：禁用后需重启系统才能生效"
    fi
fi

# ====== 验证配置 ======
echo -e "\n===== Nouveau驱动状态验证 ====="
if grep -q "blacklist nouveau" /etc/modprobe.d/blacklist.conf; then
    echo "当前状态: 禁用 (已添加黑名单)"
else
    echo "当前状态: 启用 (黑名单已移除)"
fi

echo -e "\n📌 重要提示:"
echo "1. 禁用Nouveau后，必须重启系统才能生效"
echo "2. 重启后，NVIDIA驱动才能正确加载"
echo "3. 启用Nouveau后，需重启系统才能生效"
echo "4. 配置文件备份: $backup_file"
echo "========================"

# ====== 执行建议 ======
echo -e "\n✅ Nouveau驱动配置完成！"
if [ "$enable_nouveau" = true ]; then
    echo "请执行: sudo reboot 以启用Nouveau驱动"
else
    echo "请执行: sudo reboot 以禁用Nouveau驱动并安装NVIDIA驱动"
fi

echo "后续步骤："
echo "1. 重启系统后安装NVIDIA驱动"
echo "2. 安装命令示例: sudo ubuntu-drivers autoinstall"