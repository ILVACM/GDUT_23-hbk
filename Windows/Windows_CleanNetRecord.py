#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Windows 注册表网络历史记录清理工具
=====================================
功能：清除 Windows 系统注册表中冗余的历史网络配置文件记录
作者：自动化脚本生成
版本：1.0.0
日期：2024

安全警告：
- 本脚本涉及注册表操作，请谨慎使用
- 建议在操作前创建系统还原点
- 请确保以管理员权限运行

使用方法：
- 右键点击脚本，选择"以管理员身份运行"
- 或在管理员权限的命令提示符/PowerShell中执行
"""

import winreg
import ctypes
import sys
import os
import datetime
import shutil
from typing import Dict, List, Tuple, Optional


# ============================================================================
# 常量定义
# ============================================================================

# 注册表网络配置文件路径
NETWORK_PROFILES_REG_PATH = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\NetworkList\Profiles"

# 日志文件路径（保存在脚本所在目录）
LOG_FILE_NAME = "NetworkCleanLog.txt"

# 备份目录名称
BACKUP_DIR_NAME = "NetworkBackup"


# ============================================================================
# 日志记录模块
# ============================================================================

class OperationLogger:
    """
    操作日志记录器
    功能：记录脚本执行过程中的所有关键操作和结果
    """
    
    def __init__(self, log_dir: str):
        """
        初始化日志记录器
        
        参数：
            log_dir: 日志文件保存目录
        """
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, LOG_FILE_NAME)
        self.operation_count = 0
        
    def log(self, message: str, level: str = "INFO") -> None:
        """
        记录日志信息
        
        参数：
            message: 日志消息内容
            level: 日志级别（INFO, WARNING, ERROR, SUCCESS）
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        # 输出到控制台
        if level == "ERROR":
            print(f"❌ {message}")
        elif level == "WARNING":
            print(f"⚠️ {message}")
        elif level == "SUCCESS":
            print(f"✅ {message}")
        else:
            print(f"ℹ️ {message}")
        
        # 写入日志文件
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_entry + "\n")
        except Exception as e:
            print(f"警告：无法写入日志文件 - {e}")
    
    def log_separator(self) -> None:
        """记录分隔线，用于区分不同操作批次"""
        separator = "=" * 60
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{separator}\n")
        except Exception:
            pass


# ============================================================================
# 管理员权限检查模块
# ============================================================================

def is_admin() -> bool:
    """
    检查当前进程是否以管理员权限运行
    
    返回值：
        True: 已具有管理员权限
        False: 未具有管理员权限
    
    实现原理：
        通过 Windows API 调用检查当前进程的令牌权限级别
    """
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except Exception:
        return False


def check_admin_privilege(logger: OperationLogger) -> bool:
    """
    检查并验证管理员权限
    
    参数：
        logger: 日志记录器实例
    
    返回值：
        True: 权限验证通过
        False: 权限验证失败
    
    说明：
        如果权限不足，将输出详细的错误提示并终止程序
    """
    logger.log("正在检查管理员权限...", "INFO")
    
    if is_admin():
        logger.log("管理员权限验证通过", "SUCCESS")
        return True
    else:
        logger.log("权限验证失败：当前未以管理员身份运行", "ERROR")
        print("\n" + "=" * 60)
        print("⚠️  权限不足警告")
        print("=" * 60)
        print("本脚本需要管理员权限才能操作注册表。")
        print("\n请使用以下方式之一运行本脚本：")
        print("  1. 右键点击脚本 → 选择'以管理员身份运行'")
        print("  2. 在管理员权限的命令提示符中执行")
        print("  3. 在管理员权限的 PowerShell 中执行")
        print("=" * 60)
        return False


# ============================================================================
# 注册表操作模块
# ============================================================================

def get_network_profiles(logger: OperationLogger) -> Dict[str, str]:
    """
    从注册表读取所有网络配置文件信息
    
    参数：
        logger: 日志记录器实例
    
    返回值：
        字典：{子项名称: ProfileName值}
        例如：{"{GUID-1}": "网络1", "{GUID-2}": "网络2"}
    
    实现流程：
        1. 打开注册表指定路径
        2. 枚举所有子项
        3. 读取每个子项的 ProfileName 值
        4. 建立映射关系并返回
    
    异常处理：
        - 注册表路径不存在
        - 权限不足
        - ProfileName 值不存在或格式错误
    """
    profiles = {}
    
    logger.log(f"正在读取注册表路径：HKEY_LOCAL_MACHINE\\{NETWORK_PROFILES_REG_PATH}", "INFO")
    
    try:
        # 打开注册表项
        # KEY_READ 权限用于读取操作
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            NETWORK_PROFILES_REG_PATH,
            0,
            winreg.KEY_READ
        )
        
        # 枚举所有子项
        subkey_index = 0
        while True:
            try:
                # 获取子项名称
                subkey_name = winreg.EnumKey(key, subkey_index)
                
                # 检查是否为GUID格式的子项（以 "{" 开头和 "}" 结尾）
                if subkey_name.startswith("{") and subkey_name.endswith("}"):
                    try:
                        # 打开子项读取 ProfileName
                        subkey = winreg.OpenKey(key, subkey_name)
                        try:
                            # 读取 ProfileName 值
                            # QueryValueEx 返回 (值, 类型)
                            profile_name, _ = winreg.QueryValueEx(subkey, "ProfileName")
                            profiles[subkey_name] = str(profile_name)
                        except FileNotFoundError:
                            # ProfileName 值不存在，使用子项名称作为默认值
                            profiles[subkey_name] = f"[未命名配置] {subkey_name}"
                            logger.log(f"子项 {subkey_name} 缺少 ProfileName 值", "WARNING")
                        finally:
                            winreg.CloseKey(subkey)
                    except PermissionError:
                        logger.log(f"无法访问子项 {subkey_name}：权限不足", "WARNING")
                    except Exception as e:
                        logger.log(f"读取子项 {subkey_name} 时发生错误：{e}", "WARNING")
                
                subkey_index += 1
                
            except OSError:
                # 枚举完成，没有更多子项
                break
        
        winreg.CloseKey(key)
        
        logger.log(f"成功读取 {len(profiles)} 个网络配置文件", "SUCCESS")
        return profiles
        
    except FileNotFoundError:
        logger.log("注册表路径不存在，可能系统版本不支持", "ERROR")
        return {}
    except PermissionError:
        logger.log("无法访问注册表：权限不足", "ERROR")
        return {}
    except Exception as e:
        logger.log(f"读取注册表时发生未知错误：{e}", "ERROR")
        return {}


def backup_registry_key(subkey_name: str, profile_name: str, 
                        backup_dir: str, logger: OperationLogger) -> bool:
    """
    备份单个注册表项到文件
    
    参数：
        subkey_name: 子项名称（GUID格式）
        profile_name: 配置文件名称
        backup_dir: 备份目录路径
        logger: 日志记录器实例
    
    返回值：
        True: 备份成功
        False: 备份失败
    
    说明：
        使用 reg export 命令导出注册表项
        备份文件命名格式：日期时间_GUID.reg
    """
    try:
        # 创建备份目录
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        
        # 生成备份文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_profile_name = "".join(c for c in profile_name if c.isalnum() or c in (' ', '-', '_'))
        backup_file = os.path.join(
            backup_dir, 
            f"{timestamp}_{subkey_name}.reg"
        )
        
        # 使用 reg export 命令导出注册表项
        # 注意：reg 命令的路径格式需要使用 HKLM 而非 HKEY_LOCAL_MACHINE
        reg_path = f"HKLM\\{NETWORK_PROFILES_REG_PATH}\\{subkey_name}"
        
        # 构建并执行导出命令
        export_cmd = f'reg export "{reg_path}" "{backup_file}" /y'
        result = os.system(export_cmd)
        
        if result == 0:
            logger.log(f"已备份注册表项 {subkey_name} 到 {backup_file}", "SUCCESS")
            return True
        else:
            logger.log(f"备份注册表项 {subkey_name} 失败，错误码：{result}", "WARNING")
            return False
            
    except Exception as e:
        logger.log(f"备份过程中发生错误：{e}", "WARNING")
        return False


def delete_registry_subkey(subkey_name: str, logger: OperationLogger) -> bool:
    """
    删除指定的注册表子项
    
    参数：
        subkey_name: 要删除的子项名称（GUID格式）
        logger: 日志记录器实例
    
    返回值：
        True: 删除成功
        False: 删除失败
    
    安全机制：
        1. 验证子项名称格式（必须为GUID格式）
        2. 使用 KEY_ALL_ACCESS 权限确保可删除
        3. 完整的异常捕获和处理
    
    风险说明：
        注册表删除操作不可逆，请确保已备份重要数据
    """
    # 安全校验：确保子项名称为GUID格式
    if not (subkey_name.startswith("{") and subkey_name.endswith("}")):
        logger.log(f"无效的子项名称格式：{subkey_name}，跳过删除", "ERROR")
        return False
    
    try:
        # 打开父项，获取完全访问权限
        parent_key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            NETWORK_PROFILES_REG_PATH,
            0,
            winreg.KEY_ALL_ACCESS
        )
        
        try:
            # 执行删除操作
            winreg.DeleteKey(parent_key, subkey_name)
            logger.log(f"成功删除注册表项：{subkey_name}", "SUCCESS")
            return True
            
        except FileNotFoundError:
            logger.log(f"注册表项不存在：{subkey_name}", "WARNING")
            return False
        except PermissionError:
            logger.log(f"删除失败，权限不足：{subkey_name}", "ERROR")
            return False
        finally:
            winreg.CloseKey(parent_key)
            
    except Exception as e:
        logger.log(f"删除注册表项 {subkey_name} 时发生错误：{e}", "ERROR")
        return False


# ============================================================================
# 用户交互模块
# ============================================================================

def display_profiles(profiles: Dict[str, str]) -> None:
    """
    在终端显示所有网络配置文件列表
    
    参数：
        profiles: 网络配置文件字典 {子项名称: ProfileName}
    
    显示格式：
        序号  ProfileName           子项名称
        ─────────────────────────────────────────
        1     网络1                 {GUID-1}
        2     网络2                 {GUID-2}
    """
    print("\n" + "=" * 70)
    print("📋 检测到的网络配置文件列表")
    print("=" * 70)
    print(f"{'序号':<6} {'ProfileName':<30} {'子项名称'}")
    print("-" * 70)
    
    for index, (subkey, profile_name) in enumerate(profiles.items(), start=1):
        # 截断过长的名称以保持显示整齐
        display_name = profile_name[:28] + "..." if len(profile_name) > 30 else profile_name
        print(f"{index:<6} {display_name:<30} {subkey}")
    
    print("=" * 70)
    print(f"共计：{len(profiles)} 个网络配置文件\n")


def get_selection_mode(logger: OperationLogger) -> str:
    """
    获取用户选择的操作模式
    
    参数：
        logger: 日志记录器实例
    
    返回值：
        "forward": 正向选择模式（选择要删除的项）
        "reverse": 反向选择模式（选择要保留的项）
    
    说明：
        正向选择适合删除少量项
        反向选择适合保留少量项（删除大量项）
    """
    print("\n📌 请选择操作模式：")
    print("  [1] 正向选择 - 选择要删除的网络配置文件")
    print("  [2] 反向选择 - 选择要保留的网络配置文件（其余全部删除）")
    
    while True:
        choice = input("\n请输入选择 [1/2]: ").strip()
        
        if choice == "1":
            logger.log("用户选择：正向选择模式", "INFO")
            return "forward"
        elif choice == "2":
            logger.log("用户选择：反向选择模式", "INFO")
            return "reverse"
        else:
            print("❌ 无效输入，请输入 1 或 2")


def parse_user_input(input_str: str, max_count: int, logger: OperationLogger) -> Optional[List[int]]:
    """
    解析用户输入的编号列表
    
    参数：
        input_str: 用户输入的字符串
        max_count: 最大有效编号
        logger: 日志记录器实例
    
    返回值：
        解析后的编号列表，如 [1, 3, 5, 6, 7]
        解析失败返回 None
    
    支持格式：
        - 单个编号：5
        - 多个编号：1,3,5
        - 连续范围：1-5
        - 混合格式：1,3,5-8,10
    
    示例：
        输入 "1,3,5-7" → 返回 [1, 3, 5, 6, 7]
        输入 "1-5" → 返回 [1, 2, 3, 4, 5]
    """
    try:
        numbers = set()
        
        # 分割输入（支持逗号和空格分隔）
        parts = input_str.replace(" ", ",").split(",")
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            if "-" in part:
                # 处理范围格式，如 "1-5"
                range_parts = part.split("-")
                if len(range_parts) != 2:
                    logger.log(f"无效的范围格式：{part}", "WARNING")
                    return None
                
                start = int(range_parts[0].strip())
                end = int(range_parts[1].strip())
                
                if start > end:
                    logger.log(f"范围起始值不能大于结束值：{part}", "WARNING")
                    return None
                
                # 添加范围内的所有编号
                for num in range(start, end + 1):
                    numbers.add(num)
            else:
                # 处理单个编号
                numbers.add(int(part))
        
        # 验证编号范围
        invalid_numbers = [n for n in numbers if n < 1 or n > max_count]
        if invalid_numbers:
            logger.log(f"编号超出有效范围 (1-{max_count})：{invalid_numbers}", "WARNING")
            return None
        
        # 排序并返回
        return sorted(list(numbers))
        
    except ValueError as e:
        logger.log(f"输入格式错误：{e}", "WARNING")
        return None


def get_user_selections(total_count: int, logger: OperationLogger) -> List[int]:
    """
    获取用户选择的编号列表
    
    参数：
        total_count: 可选项总数
        logger: 日志记录器实例
    
    返回值：
        用户选择的编号列表
    """
    print(f"\n📝 请输入要选择的编号（共 {total_count} 项可选）")
    print("支持格式：")
    print("  - 单个编号：5")
    print("  - 多个编号：1,3,5")
    print("  - 连续范围：1-5")
    print("  - 混合格式：1,3,5-8,10")
    
    while True:
        user_input = input("\n请输入编号: ").strip()
        
        if not user_input:
            print("❌ 输入不能为空，请重新输入")
            continue
        
        result = parse_user_input(user_input, total_count, logger)
        
        if result is not None:
            if len(result) == 0:
                print("❌ 未选择任何项，请重新输入")
                continue
            
            print(f"\n已选择 {len(result)} 项：{result}")
            return result
        
        print("❌ 输入格式无效，请重新输入")


def confirm_operation(items_to_delete: List[Tuple[str, str]], 
                      items_to_keep: List[Tuple[str, str]],
                      mode: str, logger: OperationLogger) -> bool:
    """
    二次确认删除操作
    
    参数：
        items_to_delete: 待删除项列表 [(子项名称, ProfileName), ...]
        items_to_keep: 待保留项列表 [(子项名称, ProfileName), ...]
        mode: 操作模式
        logger: 日志记录器实例
    
    返回值：
        True: 用户确认执行
        False: 用户取消操作
    
    安全机制：
        显示详细的待删除项列表，要求用户明确确认
    """
    print("\n" + "=" * 70)
    print("⚠️  重要：删除操作确认")
    print("=" * 70)
    
    print(f"\n操作模式：{'正向选择（删除选中项）' if mode == 'forward' else '反向选择（保留选中项）'}")
    
    print(f"\n将要删除的项（共 {len(items_to_delete)} 个）：")
    print("-" * 70)
    for index, (subkey, profile_name) in enumerate(items_to_delete, start=1):
        display_name = profile_name[:28] + "..." if len(profile_name) > 30 else profile_name
        print(f"  {index:<4} {display_name:<30} {subkey}")
    
    print(f"\n将要保留的项（共 {len(items_to_keep)} 个）：")
    print("-" * 70)
    for index, (subkey, profile_name) in enumerate(items_to_keep, start=1):
        display_name = profile_name[:28] + "..." if len(profile_name) > 30 else profile_name
        print(f"  {index:<4} {display_name:<30} {subkey}")
    
    print("\n" + "=" * 70)
    print("⚠️  警告：注册表删除操作不可逆！")
    print("   建议在操作前已创建系统还原点")
    print("=" * 70)
    
    while True:
        confirm = input("\n确认执行删除操作？[Y/N]: ").strip().upper()
        
        if confirm == "Y":
            logger.log("用户确认执行删除操作", "INFO")
            return True
        elif confirm == "N":
            logger.log("用户取消删除操作", "WARNING")
            return False
        else:
            print("❌ 无效输入，请输入 Y 或 N")


# ============================================================================
# 核心业务逻辑模块
# ============================================================================

def calculate_items_to_delete(
    profiles: Dict[str, str],
    selected_indices: List[int],
    mode: str
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    根据用户选择计算待删除项和待保留项
    
    参数：
        profiles: 网络配置文件字典
        selected_indices: 用户选择的编号列表
        mode: 操作模式 ("forward" 或 "reverse")
    
    返回值：
        (待删除项列表, 待保留项列表)
        每个列表元素为 (子项名称, ProfileName) 元组
    
    逻辑说明：
        正向模式：选中项为待删除项
        反向模式：未选中项为待删除项
    """
    # 将字典转换为带索引的列表
    profile_list = list(profiles.items())
    
    # 计算选中项和未选中项的索引集合
    selected_set = set(selected_indices)
    all_indices = set(range(1, len(profile_list) + 1))
    
    if mode == "forward":
        # 正向选择：选中项为待删除项
        delete_indices = selected_set
        keep_indices = all_indices - selected_set
    else:
        # 反向选择：未选中项为待删除项
        delete_indices = all_indices - selected_set
        keep_indices = selected_set
    
    # 构建结果列表
    items_to_delete = [
        profile_list[i - 1] for i in sorted(delete_indices)
    ]
    items_to_keep = [
        profile_list[i - 1] for i in sorted(keep_indices)
    ]
    
    return items_to_delete, items_to_keep


def execute_deletion(
    items_to_delete: List[Tuple[str, str]],
    backup_dir: str,
    logger: OperationLogger
) -> Tuple[int, int, List[str]]:
    """
    执行注册表删除操作
    
    参数：
        items_to_delete: 待删除项列表 [(子项名称, ProfileName), ...]
        backup_dir: 备份目录路径
        logger: 日志记录器实例
    
    返回值：
        (成功数量, 失败数量, 失败项列表)
    
    流程：
        1. 遍历待删除项
        2. 尝试备份每个项
        3. 执行删除操作
        4. 记录结果
    """
    success_count = 0
    fail_count = 0
    failed_items = []
    
    logger.log(f"开始执行删除操作，共 {len(items_to_delete)} 项", "INFO")
    logger.log_separator()
    
    for subkey_name, profile_name in items_to_delete:
        logger.log(f"正在处理：{profile_name} ({subkey_name})", "INFO")
        
        # 尝试备份
        backup_result = backup_registry_key(subkey_name, profile_name, backup_dir, logger)
        if not backup_result:
            logger.log(f"备份失败，但仍将尝试删除 {subkey_name}", "WARNING")
        
        # 执行删除
        delete_result = delete_registry_subkey(subkey_name, logger)
        
        if delete_result:
            success_count += 1
        else:
            fail_count += 1
            failed_items.append(f"{profile_name} ({subkey_name})")
    
    return success_count, fail_count, failed_items


def generate_report(
    total_count: int,
    success_count: int,
    fail_count: int,
    failed_items: List[str],
    remaining_count: int,
    logger: OperationLogger
) -> None:
    """
    生成并显示操作结果报告
    
    参数：
        total_count: 预期删除项总数
        success_count: 成功删除数量
        fail_count: 删除失败数量
        failed_items: 失败项列表
        remaining_count: 剩余配置文件数量
        logger: 日志记录器实例
    """
    logger.log_separator()
    
    print("\n" + "=" * 70)
    print("📊 操作结果报告")
    print("=" * 70)
    
    print(f"\n📈 统计信息：")
    print(f"  • 预期删除项数量：{total_count}")
    print(f"  • 成功删除数量：{success_count}")
    print(f"  • 删除失败数量：{fail_count}")
    print(f"  • 剩余配置文件数量：{remaining_count}")
    
    if fail_count > 0:
        print(f"\n❌ 删除失败的项：")
        for item in failed_items:
            print(f"  • {item}")
    
    print("\n" + "-" * 70)
    
    if fail_count == 0:
        print("✅ 所有操作已成功完成！")
        logger.log("所有删除操作成功完成", "SUCCESS")
    else:
        print("⚠️  部分操作失败，请检查日志获取详细信息")
        logger.log(f"部分操作失败：{fail_count} 项", "WARNING")
    
    print("\n💡 后续建议：")
    print("  1. 如需恢复，请使用备份目录中的 .reg 文件")
    print("  2. 建议重启计算机以确保更改生效")
    print("  3. 如遇网络问题，请使用系统还原点恢复")
    
    print("=" * 70)


# ============================================================================
# 主程序入口
# ============================================================================

def main():
    """
    主程序入口函数
    
    执行流程：
        1. 初始化日志记录器
        2. 检查管理员权限
        3. 读取注册表网络配置文件
        4. 显示配置文件列表
        5. 获取用户操作模式选择
        6. 获取用户选择的编号
        7. 计算待删除项
        8. 二次确认
        9. 执行删除操作
        10. 生成操作报告
    """
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 初始化日志记录器
    logger = OperationLogger(script_dir)
    
    # 记录脚本启动
    logger.log_separator()
    logger.log("Windows 注册表网络历史记录清理工具启动", "INFO")
    logger.log(f"脚本路径：{script_dir}", "INFO")
    
    # 设置备份目录
    backup_dir = os.path.join(script_dir, BACKUP_DIR_NAME)
    
    # 步骤1：检查管理员权限
    print("\n" + "=" * 70)
    print("🔍 步骤 1/6：检查管理员权限")
    print("=" * 70)
    
    if not check_admin_privilege(logger):
        logger.log("程序终止：权限不足", "ERROR")
        input("\n按回车键退出...")
        sys.exit(1)
    
    # 步骤2：读取注册表网络配置文件
    print("\n" + "=" * 70)
    print("📂 步骤 2/6：读取注册表网络配置文件")
    print("=" * 70)
    
    profiles = get_network_profiles(logger)
    
    if not profiles:
        logger.log("未找到任何网络配置文件，程序终止", "WARNING")
        print("\n未检测到任何网络配置文件，无需清理。")
        input("\n按回车键退出...")
        sys.exit(0)
    
    # 步骤3：显示配置文件列表
    print("\n" + "=" * 70)
    print("📋 步骤 3/6：显示网络配置文件列表")
    print("=" * 70)
    
    display_profiles(profiles)
    
    # 步骤4：获取用户操作模式选择
    print("\n" + "=" * 70)
    print("⚙️  步骤 4/6：选择操作模式")
    print("=" * 70)
    
    mode = get_selection_mode(logger)
    
    # 步骤5：获取用户选择的编号
    print("\n" + "=" * 70)
    print("📝 步骤 5/6：选择配置文件")
    print("=" * 70)
    
    selected_indices = get_user_selections(len(profiles), logger)
    
    # 计算待删除项和待保留项
    items_to_delete, items_to_keep = calculate_items_to_delete(
        profiles, selected_indices, mode
    )
    
    # 步骤6：二次确认
    print("\n" + "=" * 70)
    print("⚠️  步骤 6/6：确认操作")
    print("=" * 70)
    
    if not confirm_operation(items_to_delete, items_to_keep, mode, logger):
        print("\n操作已取消。")
        logger.log("用户取消操作，程序退出", "INFO")
        input("\n按回车键退出...")
        sys.exit(0)
    
    # 执行删除操作
    print("\n" + "=" * 70)
    print("🔧 正在执行删除操作...")
    print("=" * 70)
    
    success_count, fail_count, failed_items = execute_deletion(
        items_to_delete, backup_dir, logger
    )
    
    # 获取剩余配置文件数量
    remaining_profiles = get_network_profiles(logger)
    remaining_count = len(remaining_profiles)
    
    # 生成操作报告
    generate_report(
        len(items_to_delete),
        success_count,
        fail_count,
        failed_items,
        remaining_count,
        logger
    )
    
    # 记录程序结束
    logger.log("程序执行完毕", "INFO")
    logger.log_separator()
    
    input("\n按回车键退出...")


if __name__ == "__main__":
    main()
