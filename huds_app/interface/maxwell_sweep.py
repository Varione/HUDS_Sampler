import json
import os
import sys
import time
import subprocess
import argparse
from win32com.client import Dispatch

# 设置控制台编码
if sys.platform == 'win32' and sys.stdout is not None:
    sys.stdout.reconfigure(encoding='utf-8')
def _match_aedt_file(aedt_files, base_dir):
    """Match correct .aedt file when multiple exist in directory."""
    if len(aedt_files) == 1:
        return aedt_files[0]
    # Try matching by directory name (project folder)
    dir_name = os.path.basename(os.path.normpath(base_dir))
    for f in aedt_files:
        if f.startswith(dir_name):
            return f
    return aedt_files[0]


def find_aedt_registry():
    """从注册表查找 AEDT 安装路径"""
    import winreg

    def _enum_keys(base_path):
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base_path) as key:
                i = 0
                while True:
                    try:
                        yield winreg.EnumKey(key, i)
                        i += 1
                    except (FileNotFoundError, OSError):
                        break
        except (FileNotFoundError, OSError):
            pass

    def _query_value(key_path, value_name=""):
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                return winreg.QueryValueEx(key, value_name)[0]
        except (FileNotFoundError, OSError):
            return None

    def _find_exe(install_dir):
        if not install_dir or not os.path.isdir(install_dir):
            return None
        exe = os.path.join(install_dir, "Win64", "ansysedt.exe")
        if os.path.exists(exe):
            return exe
        for root, dirs, files in os.walk(install_dir):
            for f in files:
                if f == "ansysedt.exe":
                    return os.path.join(root, f)
        return None

    # 经典 Ansoft 注册表路径
    for base in [r"SOFTWARE\WOW6432Node\Ansoft", r"SOFTWARE\Ansoft"]:
        for subkey in _enum_keys(base):
            val = _query_value(f"{base}\\{subkey}\\InstallationDirectory")
            if val:
                found = _find_exe(val)
                if found:
                    return found

    # ANSYS, Inc. 注册表路径 (2021+)
    for base in [r"SOFTWARE\WOW6432Node\ANSYS, Inc.", r"SOFTWARE\ANSYS, Inc."]:
        for subkey in _enum_keys(base):
            val = _query_value(f"{base}\\{subkey}")
            if val:
                found = _find_exe(val)
                if found:
                    return found

    # 检查 ANSYS Electromagnetics 下的版本子键
    for base in [r"SOFTWARE\WOW6432Node\ANSYS, Inc.\ANSYS Electromagnetics",
                 r"SOFTWARE\ANSYS, Inc.\ANSYS Electromagnetics"]:
        for version_key in _enum_keys(base):
            val = _query_value(f"{base}\\{version_key}")
            if val:
                found = _find_exe(val)
                if found:
                    return found

    return None


def find_aedt_scan():
    """扫描常见安装路径"""
    import winreg

    drives = set()
    try:
        home_drive = winreg.QueryValueEx(
            winreg.HKEY_LOCAL_MACHINE,
            "SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment",
            "HOMEDRIVE",
        )[0]
        if home_drive:
            drives.add(home_drive)
    except Exception:
        pass

    for d in ["C:", "D:", "E:", "F:", "G:"]:
        try:
            if os.path.isdir(d):
                drives.add(d)
        except OSError:
            continue

    for drive in sorted(drives):
        for ver in [
            "21.1", "22.1", "22.2", "23.1", "23.2", "24.1", "24.2",
            "25.1", "25.2", "26.1",
        ]:
            for base in ["ansys", "Ansys"]:
                for product in [f"AnsysEM{ver}", f"EM{ver}"]:
                    path = os.path.join(drive, base, product, "Win64", "ansysedt.exe")
                    if os.path.exists(path):
                        return path
    return None


def find_aedt_exe():
    """自动查找 AEDT 可执行文件 (注册表优先，回退到全盘扫描)"""
    path = find_aedt_registry()
    if path:
        return path
    return find_aedt_scan()


def ensure_aedt_running():
    """确保 AEDT 已运行，否则自动启动"""
    try:
        import psutil
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] and 'ansysedt' in proc.info['name'].lower():
                return True
    except ImportError:
        pass

    print("AEDT 未运行，正在自动查找并启动...")
    aedt_path = find_aedt_exe()
    if not aedt_path:
        raise FileNotFoundError(
            "找不到 AEDT 可执行文件。请手动启动 AEDT 后再运行，或设置环境变量 AEDT_PATH。"
        )
    subprocess.Popen([aedt_path])
    print(f"找到 AEDT: {aedt_path}")
    print("等待 AEDT 初始化（约 10 秒）...")
    time.sleep(10)
    return True


def connect_aedt():
    """连接 AEDT (自动检测版本)"""
    print("正在连接 AEDT...")

    ensure_aedt_running()

    prog_ids = [
        'Ansoft.ElectronicsDesktop.2021.1',
        'Ansoft.ElectronicsDesktop.2022.1',
        'Ansoft.ElectronicsDesktop.2023.1',
        'Ansoft.ElectronicsDesktop.2024.1',
        'Ansoft.ElectronicsDesktop.2025.1',
    ]

    for prog_id in prog_ids:
        try:
            oApp = Dispatch(prog_id)
            oDesktop = oApp.GetAppDesktop()
            print(f"AEDT 版本: {oDesktop.GetVersion()}")
            return oDesktop, oApp
        except Exception:
            continue

    raise RuntimeError("无法连接 AEDT，请确认已安装 Ansys Electronics Desktop")


def open_project(oDesktop, project_path):
    """打开或获取已打开的项目"""
    # Handle both file path and directory path
    if os.path.isdir(project_path):
        aedt_files = [f for f in os.listdir(project_path) if f.endswith('.aedt')]
        if not aedt_files:
            raise FileNotFoundError(f"No .aedt file found in: {project_path}")
        # Match by project name if multiple files exist
        matched = _match_aedt_file(aedt_files, project_path)
        project_path = os.path.join(project_path, matched)

    lock_file = project_path + ".lock"
    if os.path.exists(lock_file):
        os.remove(lock_file)

    # 检查是否已打开 — use exact path match first, then name match
    projects = oDesktop.GetProjects()
    base_name = os.path.basename(project_path).replace('.aedt', '')
    matched_project = None
    for i in range(projects.Count):
        p = projects(i)
        try:
            p_path = p.GetPath()
            if os.path.samefile(p_path, project_path):
                matched_project = p
                break
        except Exception:
            pass
        if p.GetName() == base_name:
            matched_project = p
            break

    if matched_project:
        print(f"项目已在内存中: {matched_project.GetName()}")
        return matched_project

    print(f"正在打开项目: {project_path}")
    oProject = oDesktop.OpenProject(project_path)
    print(f"项目名称: {oProject.GetName()}")
    return oProject


def list_designs(oProject):
    """列举设计列表"""
    designs = oProject.GetDesigns()
    print("\n设计列表:")
    for i in range(designs.Count):
        print(f"  [{i}] {designs(i).GetName()}")
    return designs


def select_design(oProject, design_name):
    """选择设计"""
    design_name = design_name.strip()
    oDesign = oProject.SetActiveDesign(design_name)
    print(f"已激活设计: {oDesign.GetName()}")
    return oDesign


def import_parametric_csv(oDesign, csv_path, setup_name="ParametricSetup1"):
    """导入参数扫描 CSV 文件"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    print(f"\n正在导入参数扫描文件: {csv_path}")
    oOpti = oDesign.GetModule("Optimetrics")

    # 删除旧的 setup（如果存在）
    try:
        oOpti.DeleteSetups([setup_name])
        print(f"已删除旧 setup: {setup_name}")
    except Exception:
        pass

    # 导入新的 CSV
    oOpti.ImportSetup("OptiParametric", ["NAME:" + setup_name, csv_path])
    print(f"参数扫描 setup 导入成功")

    return oOpti


def run_parametric_sweep(oOpti, setup_name="ParametricSetup1", progress_cb=None, progress_file=None):
    """运行参数扫描(同步),支持生命周期回调.

    Args:
        oOpti: Optimetrics module object
        setup_name: Setup name
        progress_cb: Optional callback with lifecycle events:
            ("started", total=0) on entry
            ("completed", total=1) on success
            ("failed", error_msg) on exception
        progress_file: Optional file path to write JSON progress for external polling
    """
    print(f"\n正在启动参数扫描: {setup_name}")
    print("（此过程可能需要较长时间，请耐心等待...）")

    start_time = time.time()

    # Signal started
    if progress_cb:
        progress_cb("started", 0)
    if progress_file:
        try:
            tmp_file = progress_file + ".tmp"
            with open(tmp_file, "w", encoding="utf-8") as pf:
                json.dump({"status": "running", "done": 0, "total": 0, "failed": 0}, pf)
            os.replace(tmp_file, progress_file)
        except Exception:
            pass

    try:
        oOpti.SolveSetup(setup_name)
        elapsed = time.time() - start_time
        print(f"参数扫描完成，耗时 {elapsed:.1f} 秒")

        # Signal completed
        if progress_cb:
            progress_cb("completed", 1)
        if progress_file:
            try:
                tmp_file = progress_file + ".tmp"
                with open(tmp_file, "w", encoding="utf-8") as pf:
                    json.dump({"status": "completed", "done": 1, "total": 1, "failed": 0}, pf)
                os.replace(tmp_file, progress_file)
            except Exception:
                pass
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"参数扫描失败 ({elapsed:.1f} 秒): {e}")

        # Signal failed
        if progress_cb:
            progress_cb("failed", str(e))
        if progress_file:
            try:
                tmp_file = progress_file + ".tmp"
                with open(tmp_file, "w", encoding="utf-8") as pf:
                    json.dump({"status": "failed", "error": str(e)}, pf)
                os.replace(tmp_file, progress_file)
            except Exception:
                pass

        raise


def check_sweep_status(oOpti, setup_name="ParametricSetup1"):
    """检查扫描状态"""
    setup_obj = oOpti.GetChildObject(setup_name)
    has_result = setup_obj.GetPropValue("HasResult")
    return has_result


def export_results(oRpt, output_dir):
    """导出所有结果表"""
    report_names = oRpt.GetAllReportNames()
    if not report_names:
        print("\n没有可用的结果表")
        return []

    exported_files = []
    for name in report_names:
        trace_names = oRpt.GetReportTraceNames(name)
        if not trace_names:
            print(f"\n跳过空报告 '{name}' (无 trace)")
            continue

        csv_path = os.path.join(output_dir, f"{name}.csv")
        print(f"\n正在导出报告 '{name}' 到: {csv_path}")
        oRpt.ExportToFile(name, csv_path)
        time.sleep(1)

        if os.path.exists(csv_path):
            size = os.path.getsize(csv_path)
            print(f"导出成功，文件大小: {size} bytes")
            exported_files.append(csv_path)
        else:
            print("导出失败，文件未生成")

    return exported_files


def cleanup_results(oDesign):
    """清理所有解数据"""
    print("\n正在清理所有解数据...")
    try:
        oDesign.DeleteFullVariation(["All"], False)
        print("清理完成")
    except Exception as e:
        print(f"清理跳过 (无旧数据或失败): {e}")


def run_sweep(csv_path, config_path=None, project_path=None, design_name=None, output_dir=None, progress_cb=None, progress_file=None):
    """非交互式运行参数扫描 - 供 HUDS 调用

    Args:
        csv_path: Request CSV path
        config_path: HUDS config.json path
        project_path: AEDT project file or directory
        design_name: Design name
        output_dir: Output directory for exported results
        progress_cb: Optional callback (done, total, failed, statuses) -> None for real-time progress
        progress_file: Optional file path to write JSON progress for external polling
    """
    os.environ["ANSYSLMD_LICENSE_FILE"] = "24500@licensing.hkust.edu.cn"

    oDesktop, oApp = connect_aedt()

    if not project_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_path = os.path.join(script_dir, "优化模型1.aedt")
    if not design_name:
        design_name = "ly2"
    if not output_dir:
        output_dir = os.path.dirname(project_path)

    # 转换 CSV 为 Maxwell 参数表格式
    import shutil
    maxwell_csv = os.path.join(output_dir, "ParametricSetup1_Table.csv")
    lock_file = maxwell_csv + ".lock"
    if os.path.exists(lock_file):
        os.remove(lock_file)

    if config_path:
        from huds_app.interface.maxwell import export_maxwell_table as emt
        from huds_app.core.config import load_config
        cfg = load_config(config_path)
        emt(csv_path, maxwell_csv, config=cfg)
    else:
        shutil.copy2(csv_path, maxwell_csv)

    oProject = open_project(oDesktop, project_path)
    oDesign = select_design(oProject, design_name)

    cleanup_results(oDesign)

    setup_name = "ParametricSetup1"
    oOpti = import_parametric_csv(oDesign, maxwell_csv, setup_name)

    run_parametric_sweep(oOpti, setup_name, progress_cb=progress_cb, progress_file=progress_file)

    has_result = check_sweep_status(oOpti, setup_name)
    print(f"\n扫描结果状态: {'有结果' if has_result else '无结果'}")

    if not has_result:
        raise RuntimeError("扫描未完成或无结果")

    oRpt = oDesign.GetModule("ReportSetup")
    exported = export_results(oRpt, output_dir)

    oProject.Save()

    return exported


def main():
    parser = argparse.ArgumentParser(description="Maxwell 参数扫描工具")
    parser.add_argument("--csv", type=str, help="参数扫描 CSV 文件路径")
    parser.add_argument("--config", type=str, default=None, help="HUDS config.json 路径")
    parser.add_argument("--project", type=str, default=None, help="项目文件路径 (默认: 脚本目录下优化模型1.aedt)")
    parser.add_argument("--design", type=str, default="ly2", help="设计名称")
    parser.add_argument("--output-dir", type=str, help="输出目录")
    parser.add_argument("--progress-file", type=str, default=None, help="进度文件路径 (JSON)")
    parser.add_argument("--interactive", action="store_true", help="交互式模式")

    args = parser.parse_args()

    if args.csv:
        exported = run_sweep(
            csv_path=args.csv,
            config_path=args.config,
            project_path=args.project,
            design_name=args.design,
            output_dir=args.output_dir or os.path.dirname(args.project),
            progress_file=args.progress_file,
        )
        print(f"\n流程完成！导出了 {len(exported)} 个文件")
        return

    if args.interactive:
        _run_interactive()
    else:
        # 默认交互式
        _run_interactive()


def _run_interactive():
    """交互式模式 - 保留原有功能"""
    os.environ["ANSYSLMD_LICENSE_FILE"] = "24500@licensing.hkust.edu.cn"

    oDesktop, oApp = connect_aedt()

    project_path = input("\n请输入仿真项目 (.aedt) 文件路径: ").strip().strip('"')
    if not os.path.exists(project_path):
        print(f"文件不存在: {project_path}")
        return

    oProject = open_project(oDesktop, project_path)
    designs = list_designs(oProject)

    idx = int(input("\n请输入要操作的设计编号: ").strip())
    design_name = designs(idx).GetName()
    oDesign = select_design(oProject, design_name)

    cleanup = input("\n是否清理旧结果？(y/n): ").strip().lower()
    if cleanup == "y":
        cleanup_results(oDesign)

    csv_path = input("\n请输入参数扫描 CSV 文件路径: ").strip().strip('"')

    setup_name = input("请输入 Setup 名称 (直接回车使用默认 ParametricSetup1): ").strip()
    if not setup_name:
        setup_name = "ParametricSetup1"

    oOpti = import_parametric_csv(oDesign, csv_path, setup_name)
    run_parametric_sweep(oOpti, setup_name)

    has_result = check_sweep_status(oOpti, setup_name)
    print(f"\n扫描结果状态: {'有结果' if has_result else '无结果'}")

    if not has_result:
        print("扫描未完成或无结果，无法导出")
        return

    save_fields = input("\n是否启用 SaveFields（保存场数据，导出更完整）？(y/n): ").strip().lower()
    if save_fields == "y":
        setup_obj = oOpti.GetChildObject(setup_name)
        sf = setup_obj.GetPropValue("SaveFields")
        if not sf:
            print("启用 SaveFields，需要重新运行扫描...")
            setup_obj.SetPropValue("SaveFields", True)
    run_parametric_sweep(oOpti, setup_name, progress_cb=None, progress_file=None)

    output_dir = os.path.dirname(project_path)
    print(f"\n结果将导出到: {output_dir}")

    oRpt = oDesign.GetModule("ReportSetup")
    report_names = oRpt.GetAllReportNames()
    if report_names:
        print("\n可用结果表:")
        for i, name in enumerate(report_names):
            print(f"  [{i}] {name}")

        export_all = input("\n导出所有结果表？(y/n): ").strip().lower()
        if export_all == "y":
            export_results(oRpt, output_dir)
        else:
            idx = input("请输入要导出的报告编号 (多个用逗号分隔): ").strip()
            indices = [int(x.strip()) for x in idx.split(",")]
            for i in indices:
                name = report_names[i]
                csv_path = os.path.join(output_dir, f"{name}.csv")
                oRpt.ExportToFile(name, csv_path)
                time.sleep(1)
                print(f"已导出: {csv_path}")
    else:
        print("\n没有可用的结果表，请先创建报告或启用 SaveFields 后重新扫描")

    save = input("\n是否保存项目？(y/n): ").strip().lower()
    if save == "y":
        oProject.Save()
        print("项目已保存")

    print("\n流程完成！")


if __name__ == "__main__":
    main()