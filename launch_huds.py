import os
import sys
import json
import time
import shutil
import subprocess
import numpy as np
import pandas as pd

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.environ["ANSYSLMD_LICENSE_FILE"] = os.environ.get("ANSYSLMD_LICENSE_FILE", "24500@licensing.hkust.edu.cn")

from huds_app.core.config import load_config
from huds_app.core.storage import RunState, read_csv, write_csv
from huds_app.data.schema import SAMPLE_ID_COLUMN
from huds_app.sampling.huds import run_huds_sampling
from huds_app.data.validation import import_labels as il
from huds_app.model.train import train_model


def ask(prompt, default=None):
    if default is not None:
        prompt_str = f"{prompt} [{default}]: "
    else:
        prompt_str = f"{prompt}: "
    sys.stdout.write(prompt_str)
    sys.stdout.flush()
    val = input().strip().strip('"').strip("'")
    return val if val else (default if default is not None else "")


def ask_list(prompt, default=""):
    val = ask(prompt, default)
    val = val.strip("[]()")
    return [x.strip() for x in val.split(",") if x.strip()]


def ask_floats(prompt, default=""):
    vals = ask_list(prompt, default)
    return [float(x) for x in vals]


def ask_int(prompt, default=0):
    val = ask(prompt, str(default))
    return int(val)


# ========== AEDT 自动检测与启动 ==========

print("=" * 60)
print("HUDS 主动学习")
print("=" * 60)


def connect_aedt_auto():
    """自动检测 AEDT 版本并连接"""
    from win32com.client import Dispatch
    from huds_app.interface.maxwell_sweep import find_aedt_exe, ensure_aedt_running

    # 尝试的 COM ProgID (按常见度排序)
    prog_ids = [
        'Ansoft.ElectronicsDesktop.2021.1',
        'Ansoft.ElectronicsDesktop.2022.1',
        'Ansoft.ElectronicsDesktop.2023.1',
        'Ansoft.ElectronicsDesktop.2024.1',
        'Ansoft.ElectronicsDesktop.2025.1',
    ]

    # 先尝试直接连接 (AEDT 已在运行)
    for prog_id in prog_ids:
        try:
            oApp = Dispatch(prog_id)
            oDesktop = oApp.GetAppDesktop()
            version = oDesktop.GetVersion()
            print(f"  [OK] AEDT 已连接，版本: {version}")
            return oApp, oDesktop
        except Exception:
            continue

    # 未检测到运行中的 AEDT，尝试自动启动
    print("  AEDT 未运行，正在自动查找并启动...")
    try:
        ensure_aedt_running()
    except FileNotFoundError as e:
        print(f"\n  [错误] {e}")
        sys.exit(1)

    # 等待启动后再次尝试连接
    for prog_id in prog_ids:
        try:
            oApp = Dispatch(prog_id)
            oDesktop = oApp.GetAppDesktop()
            version = oDesktop.GetVersion()
            print(f"  [OK] AEDT 已连接，版本: {version}")
            return oApp, oDesktop
        except Exception:
            continue

    print("\n  [错误] 无法连接 AEDT，请手动启动后重试。")
    sys.exit(1)


print("\n正在检查 AEDT 连接...")
oApp, oDesktop = connect_aedt_auto()
projects = oDesktop.GetProjects()
if projects.Count > 0:
    print("  已打开的项目:")
    for j in range(projects.Count):
        print(f"    [{j}] {projects(j).GetName()}")

# ========== AEDT 项目选择 ==========

print("\n--- AEDT 仿真设置 ---")
if projects.Count > 0:
    print("已打开的项目:")
    for j in range(projects.Count):
        try:
            p_path = projects(j).GetPath()
        except Exception:
            p_path = "未知路径"
        print(f"  [{j}] {projects(j).GetName()} ({p_path})")
    print(f"  [-1] 打开新的项目文件")

    proj_choice = ask_int("请选择 (使用已打开的项目)", 0)

    aedt_path = None
    oProject = None

    if proj_choice == -1:
        default_aedt = os.environ.get("AEDT_PROJECT", "")
        aedt_path = ask(
            "AEDT 项目文件 (.aedt) 路径 (支持相对路径，相对于项目根目录)",
            default_aedt if default_aedt else None
        )
        if not os.path.isabs(aedt_path):
            aedt_path = os.path.join(PROJECT_ROOT, aedt_path)
        aedt_path = os.path.normpath(aedt_path)
        if not os.path.exists(aedt_path):
            print(f"错误: 文件不存在: {aedt_path}")
            sys.exit(1)

        # 先关闭已打开的项目，避免 OpenProject 卡住
        for j in range(projects.Count):
            try:
                oDesktop.CloseProject(projects(j).GetName())
            except Exception:
                pass
        time.sleep(2)
        oProject = oDesktop.OpenProject(aedt_path)
        time.sleep(3)
    else:
        oProject = projects(proj_choice)
        aedt_path = oProject.GetPath()
        print(f"  已选择: {oProject.GetName()}")
else:
    default_aedt = os.environ.get("AEDT_PROJECT", "")
    aedt_path = ask(
        "AEDT 项目文件 (.aedt) 路径 (支持相对路径，相对于项目根目录)",
        default_aedt if default_aedt else None
    )
    if not os.path.isabs(aedt_path):
        aedt_path = os.path.join(PROJECT_ROOT, aedt_path)
    aedt_path = os.path.normpath(aedt_path)
    if not os.path.exists(aedt_path):
        print(f"错误: 文件不存在: {aedt_path}")
        sys.exit(1)
    oProject = oDesktop.OpenProject(aedt_path)
    time.sleep(3)

project_dir = os.path.dirname(aedt_path)

# List designs
designs = oProject.GetDesigns()
print("\n设计列表:")
for j in range(designs.Count):
    print(f"  [{j}] {designs(j).GetName()}")

dsgn_idx = ask_int("请选择设计编号", 0)
design_name = designs(dsgn_idx).GetName()
print(f"  已选择设计: {design_name}")

# ========== 配置向导 ==========

print("\n" + "=" * 60)
print("配置向导")
print("=" * 60)

project_name = time.strftime("%Y%m%d_%H%M%S")
print(f"  运行目录: runs/{project_name}")

print("\n--- 扫参变量 ---")
var_names = ask_list("变量名 (逗号分隔)", "v")
var_mins = ask_floats("变量最小值 (逗号分隔)", "100")
var_maxs = ask_floats("变量最大值 (逗号分隔)", "500")
var_units = ask_list("变量单位 (逗号分隔)", "km_per_hour")

if len(var_names) != len(var_mins) or len(var_names) != len(var_maxs):
    print("错误: 变量名、最小值、最大值数量不匹配")
    sys.exit(1)

variables = []
for i, name in enumerate(var_names):
    variables.append({
        "name": name,
        "min": var_mins[i],
        "max": var_maxs[i],
        "sample_points": 60,
        "unit": var_units[i] if i < len(var_units) else ""
    })

print("\n--- 输出变量 ---")
oDesign = oProject.SetActiveDesign(design_name)
oRpt = oDesign.GetModule("ReportSetup")
report_names = oRpt.GetAllReportNames()

if report_names:
    print("已创建的报告:")
    for j, name in enumerate(report_names):
        trace_names = oRpt.GetReportTraceNames(name)
        if trace_names:
            traces_str = ", ".join(list(trace_names)[:3])
            if len(trace_names) > 3:
                traces_str += f"... (+{len(trace_names)-3})"
            print(f"  [{j}] {name} (traces: {traces_str})")
        else:
            print(f"  [{j}] {name} (空)")
    print(f"  [-1] 手动输入")

    rpt_choice = ask_int("请选择报告 (逗号分隔)", "0")
    if rpt_choice == -1:
        output_names = ask_list("仿真输出变量名 (逗号分隔)", "peak_force_y,peak_force_z")
    else:
        chosen_indices = [int(x.strip()) for x in str(rpt_choice).split(",")]
        output_names = [report_names[i] for i in chosen_indices if 0 <= i < len(report_names)]
else:
    print("未找到已创建的报告，请手动输入:")
    output_names = ask_list("仿真输出变量名 (逗号分隔)", "peak_force_y,peak_force_z")

steady_pct = float(ask("稳态值提取比例 (取末尾 X% 数据求均值，0.2=最后20%)", "0.2"))

total_samples = ask_int("候选池总样本数", 600)

print("\n--- 训练设置 ---")
initial_train = ask_int("初始训练样本数", 20)
sample_per_step = ask_int("每步新增样本数", 15)
max_steps = ask_int("最大迭代轮数", 3)
epochs = ask_int("每步训练 epoch 数", 200)
device = ask("计算设备 (cpu/cuda)", "cpu")

config = {
    "project_name": project_name,
    "random_seed": 42,
    "aedt_project_path": aedt_path,
    "design_name": design_name,
    "variables": variables,
    "candidate_pool": {"total_samples": total_samples},
    "split": {
        "train_split": 0.8,
        "val_split": 0.1,
        "test_split": 0.1
    },
    "model": {
        "model_type": "vector_to_vector",
        "output_names": output_names,
        "hidden_dim": 64,
        "encoder_blocks": 2,
        "dropout": 0.1
    },
    "training": {
        "initial_train_size": initial_train,
        "sample_per_step": sample_per_step,
        "max_steps": max_steps,
        "epochs_per_step": epochs,
        "batch_size": 32,
        "learning_rate": 0.001,
        "patience": 30,
        "device": device
    },
    "huds": {
        "repeat_times": 10,
        "topk_ratio": 0.6,
        "batch_size": 128,
        "use_faiss": False
    },
    "steady_state_pct": steady_pct
}

print("\n" + "=" * 60)
print("配置汇总")
print("=" * 60)
print(json.dumps(config, indent=2, ensure_ascii=False))

confirm = ask("\n确认创建? (y/n)", "y")
if confirm.lower() != "y":
    print("已取消")
    sys.exit(0)

# ========== 初始化运行 ==========

runs_dir = os.path.join(project_dir, "HUDS_runs")
run_dir = os.path.join(runs_dir, project_name)
os.makedirs(run_dir, exist_ok=True)
config_path = os.path.join(run_dir, "config.json")

# 确保文件句柄完全释放后再调用 init_run
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
time.sleep(1)

print(f"\n配置已保存到: {config_path}")

# 初始化运行状态
from huds_app.interface.workflow import init_run
init_run(config_path, run_dir)

print("\n初始状态:")
from huds_app.interface.workflow import show_status
show_status(run_dir)

# ========== 执行循环 ==========

def run_maxwell_sweep(request_csv, cfg_path, aedt_path, dsgn_name):
    print("  运行 Maxwell 仿真...")
    env = os.environ.copy()
    env["ANSYSLMD_LICENSE_FILE"] = "24500@licensing.hkust.edu.cn"
    env["PYTHONPATH"] = PROJECT_ROOT + ";" + env.get("PYTHONPATH", "")
    sweep_script = os.path.join(PROJECT_ROOT, "huds_app", "interface", "maxwell_sweep.py")
    result = subprocess.run(
        [sys.executable, sweep_script,
         "--csv", request_csv,
         "--config", cfg_path,
         "--project", aedt_path,
         "--design", dsgn_name],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        timeout=600,
        env=env,
    )
    if result.returncode != 0:
        err = result.stderr.replace('\ufffd', '?')[:2000]
        print(f"  仿真失败:\n{err}")
        out = result.stdout.replace('\ufffd', '?')[:1000]
        if out:
            print(f"  输出:\n{out}")
        raise RuntimeError("maxwell_sweep.py failed")

    output_csv = os.path.join(os.path.dirname(aedt_path), "Force Plot 1.csv")
    if os.path.exists(output_csv):
        size = os.path.getsize(output_csv)
        print(f"  已导出: {output_csv} ({size} bytes)")
        return output_csv
    raise RuntimeError("仿真结果文件不存在")


def extract_labels(force_csv, step, run_dir, cfg):
    request_path = os.path.join(run_dir, "requests", f"train_step_{step:03d}_request.csv")
    df = pd.read_csv(force_csv)
    request_df = read_csv(request_path)

    var_name = cfg.variables[0].name if cfg.variables else "v"
    steady_pct = getattr(cfg, "steady_state_pct", 0.2)

    # 从 CSV 列名自动提取唯一的输出标识 (如 Force_y, Force_z)
    detected_outputs = set()
    for col in df.columns[1:]:
        if f"{var_name}='" not in col:
            continue
        prefix = col.rsplit(f" - {var_name}='", 1)[0]
        detected_outputs.add(prefix)

    out_names = sorted(detected_outputs) if detected_outputs else ["peak_force_y", "peak_force_z"]
    print(f"  检测到输出: {out_names}")
    print(f"  稳态值提取: 末尾 {steady_pct*100:.0f}% 数据均值")

    force_data = []
    for col in df.columns[1:]:
        if f"{var_name}='" not in col:
            continue
        v_str = col.split(f"{var_name}='")[1].split("'")[0]
        v_val = float(v_str.replace('km_per_hour', ''))

        # 取末尾 steady_pct 比例的数据求均值 (稳态值)
        n_total = len(df[col])
        n_steady = max(1, int(n_total * steady_pct))
        steady_mean = df[col].iloc[-n_steady:].mean()

        prefix = col.rsplit(f" - {var_name}='", 1)[0]
        force_data.append({
            "v_rounded": round(v_val, 4),
            "value": steady_mean,
            "output_name": prefix
        })

    if not force_data:
        raise RuntimeError("未能从仿真结果中提取任何数据，请检查 CSV 列名格式")

    force_df = pd.DataFrame(force_data)
    pivoted = force_df.pivot(index="v_rounded", columns="output_name", values="value").reset_index()

    merged = request_df.copy()
    merged["v_rounded"] = merged[var_name].round(4)
    merged = merged.merge(pivoted, on="v_rounded", how="left")
    merged = merged.drop(columns=["v_rounded"])

    label_path = os.path.join(run_dir, "data", f"step{step}_labels.csv")
    write_csv(merged[[SAMPLE_ID_COLUMN, var_name] + out_names], label_path)
    return label_path, out_names


print("\n" + "=" * 60)
print("开始主动学习循环")
print("=" * 60)

cfg = load_config(config_path)
# HUDS 主动学习循环 (step 1+)
r2_history = []
for step in range(1, cfg.training.max_steps + 1):
    print(f"\n{'='*60}")
    print(f"步骤 {step}: HUDS 主动采样")
    print(f"{'='*60}")

    result = run_huds_sampling(run_dir, cfg, step)
    n_selected = len(result["selected_ids"])
    print(f"  选择了 {n_selected} 个样本")

    request_path = os.path.join(run_dir, "requests", f"train_step_{step:03d}_request.csv")
    req_df = pd.read_csv(request_path, dtype={'sample_id': str})
    print(f"  请求表已生成 ({len(req_df)} 行)")

    force_csv = run_maxwell_sweep(
        request_path, config_path, aedt_path, design_name
    )
    label_path, out_names = extract_labels(force_csv, step, run_dir, cfg)

    # 更新 config 中的 output_names 为真实检测到的列名
    if hasattr(cfg, 'model') and hasattr(cfg.model, 'output_names'):
        old_out = cfg.model.output_names
        cfg.model.output_names = out_names
        if old_out != out_names:
            print(f"  更新 output_names: {old_out} -> {out_names}")
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = json.load(f)
            full_config["model"]["output_names"] = out_names
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(full_config, f, indent=2, ensure_ascii=False)
            cfg = load_config(config_path)

    imported = il(run_dir, step, label_path, overwrite=True)
    print(f"  导入了 {imported} 条标签")

    _last_train_percent = [0]
    def progress_cb(msg, percent):
        if percent - _last_train_percent[0] >= 10 or percent == 100:
            print(f"  [Train] {msg}")
            _last_train_percent[0] = percent

    metrics = train_model(run_dir, cfg, progress_cb=progress_cb)
    r2 = metrics.get("val_r2_avg", float("nan"))
    r2_history.append(r2)
    r2_str = f"{r2:.4f}" if not np.isnan(r2) else "N/A (样本不足)"
    print(f"  训练完成: val_r2_avg = {r2_str}")

    print(f"\n{'='*60}")
    print(f"Step {step} 完成后状态")
    print(f"{'='*60}")
    show_status(run_dir)

print(f"\n{'='*60}")
print("全流程完成")
print(f"{'='*60}")
valid_r2 = [r for r in r2_history if not np.isnan(r)]
if len(valid_r2) >= 2:
    print(f"R2 变化: {valid_r2[0]:.4f} -> {valid_r2[-1]:.4f}")
elif valid_r2:
    print(f"最终 R2: {valid_r2[0]:.4f}")
else:
    print("R2 不可用 (验证集样本不足)")
