"""Summarize the fixed two-pass paired benchmark artifacts."""
import collections
import csv
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parents[1]
folder = root / 'artifacts/efficiency_260909'
data = {name: [json.loads((folder/f'{name}_r{i}.json').read_text()) for i in [1,2]]
        for name in ['3dlrf','wcbr']}
assert all(len(d['rows']) == 100 for ds in data.values() for d in ds)
assert len({tuple(d['indices']) for ds in data.values() for d in ds}) == 1
summary = {}
for name, ds in data.items():
    rows = [row for d in ds for row in d['rows']]
    stats = {}
    for key in ['load_ms','forward_wall_ms','forward_cuda_event_ms','postprocess_ms','detector_ms','pipeline_ms']:
        vals = [row[key] for row in rows]
        # Linear percentile interpolation matches numpy.percentile default.
        ordered = sorted(vals); q=(len(vals)-1)*.95; lo=int(q)
        stats[key] = dict(mean=statistics.mean(vals),median=statistics.median(vals),
                         p95=ordered[lo]+(ordered[lo+1]-ordered[lo])*(q-lo),
                         repeats=[d['stats'][key]['mean'] for d in ds])
    summary[name] = dict(stats=stats,params_m=ds[0]['params_m'],
        peak_allocated_gib=max(d['peak_allocated_gib'] for d in ds),
        peak_reserved_gib=max(d['peak_reserved_gib'] for d in ds))
with (folder/'per_frame.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=['model','repeat']+list(data['wcbr'][0]['rows'][0]))
    writer.writeheader()
    for name,ds in data.items():
        for repeat,d in enumerate(ds,1):
            writer.writerows(dict(model=name,repeat=repeat,**row) for row in d['rows'])
(folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
def mean(name,key): return summary[name]['stats'][key]['mean']
lines=['# WCBR / 3D-LRF 实测推理效率','',
       '运行标识 efficiency_260909；启动于2026-09-09，完成时间见 runs.json。只读参考 dec_con_asf/benchmark_inference_speed.py，未修改该项目或运行其模型。','',
       '## 固定协议','',
       '- 同一空闲物理GPU2：NVIDIA RTX A6000；主机双路Xeon Gold 5218，PyTorch线程数32。',
       '- PyTorch 1.10.1+cu113 / CUDA 11.3；FP32张量，无autocast；matmul/cudnn TF32均开启。',
       '- Batch=1，num_workers=0，no_grad + eval；strict checkpoint加载。冻结相机前端保持实际前向，未缓存图像或点云特征。',
       '- WCBR主模型：第17轮model_16.pt；3D-LRF共享协议复现：第10轮model_9.pt。',
       '- 17536帧test按索引均匀选120帧；前20帧预热，后100帧计时。两遍使用相同帧，次序为3D-LRF→WCBR→WCBR→3D-LRF。共每模型200个计时观测，但只有100个不同帧。',
       '- 这是预热后的抽样测速，不是全数据集平均或冷启动测试。未固定GPU时钟、CPU亲和性或清除OS文件缓存；反序复测用于观察顺序敏感性。',
       '- 实测100帧天气构成：'+str(dict(collections.Counter(x['weather'] for x in data['wcbr'][0]['rows'])))+'。没有overcast计时帧；不据此声称完整逐天气效率。',
       '', '## 计时范围','',
       '- forward_wall：同步后的墙钟前向，包含稀疏预处理、主机到GPU传输、相机编码、CPU KNN、骨干与检测头。另存CUDA event时间供参考，不能称其为独立GPU纯计算时间。',
       '- detector：forward_wall + 检测框解码、confidence=0.3过滤、NMS IoU=0.1、输出CPU列表。',
       '- pipeline：同步数据读取/解析/collation + detector。现有dataset会读取标注，因此这不是车载传感器接入后的延迟。',
       '- 不含模型/数据集初始化、预热、AP计算、结果文件写盘和每帧引用清理；未使用异步读取或处理流水线。',
       '- dec_con_asf参考脚本的total止于网络前向，不含上述后处理。因此不要直接把两个项目都叫端到端的FPS混表；forward可作口径参考，但抽样与模型输入也不同。',
       '', '## 两遍合并结果','',
       '| 指标 | 3D-LRF复现 | WCBR主模型 |','|---|---:|---:|']
for label,key in [('平均前向 ms','forward_wall_ms'),('平均后处理 ms','postprocess_ms'),('平均检测延迟 ms','detector_ms'),('平均读取 ms','load_ms'),('平均读取+检测 ms','pipeline_ms')]:
    lines.append(f'| {label} | {mean("3dlrf",key):.2f} | {mean("wcbr",key):.2f} |')
for label,key in [('前向FPS','forward_wall_ms'),('含后处理检测FPS','detector_ms'),('读取+检测FPS','pipeline_ms')]:
    lines.append(f'| {label} | {1000/mean("3dlrf",key):.2f} | {1000/mean("wcbr",key):.2f} |')
for label,key in [('总参数 M','params_m'),('峰值allocated GiB','peak_allocated_gib'),('峰值reserved GiB','peak_reserved_gib')]:
    lines.append(f'| {label} | {summary["3dlrf"][key]:.3f} | {summary["wcbr"][key]:.3f} |')
lines += ['', 'FPS=1000/平均毫秒，不是逐帧FPS的算术平均。显存是当前进程的PyTorch allocator统计，不包含所有驱动/第三方分配；参数数目包含加载的辅助模块。','',
          '## 重复性与尾部延迟','', '| 模型 | 遍次 | 前向均值 ms | 检测均值 ms | 检测P95 ms | 读取+检测均值 ms |','|---|---|---:|---:|---:|---:|']
for name,ds in data.items():
    for i,d in enumerate(ds,1):
        st=d['stats'];lines.append(f'| {name} | {i} | {st["forward_wall_ms"]["mean"]:.2f} | {st["detector_ms"]["mean"]:.2f} | {st["detector_ms"]["p95"]:.2f} | {st["pipeline_ms"]["mean"]:.2f} |')
overhead=(mean('wcbr','detector_ms')/mean('3dlrf','detector_ms')-1)*100
lines += ['', '## 结论与适用边界','',
          f'WCBR含后处理平均检测延迟相对复现3D-LRF变化 {overhead:+.1f}%。这是已训练checkpoint在相同帧上的实际成本差异；后处理也受各模型预测框数量影响，不可全部归因于router。',
          '读取耗时受文件缓存和存储影响；不能把较快的数据读取解释成网络结构加速。当前实现的数FPS检测速度不支持未经约束的实时部署声明，也不能外推到车载计算平台。',
          '本次未作模块级profile，不能把前向时间全部归因于CPU KNN；可确认该CPU过程在计时范围内。推理优化需单独检查数据输入、邻域搜索及实际计算耗时。',
          '', '## 复现文件','',
          f'- [测速脚本]({root / "tools/benchmark_wcbr_inference.py"})',
          f'- [汇总JSON]({folder / "summary.json"})',
          f'- [逐帧CSV]({folder / "per_frame.csv"})',
          f'- [运行记录]({folder / "runs.json"})',
          '原始每遍JSON包含checkpoint/config/script/source哈希、CUDA设置、抽样索引与逐帧耗时。']
report=root.parent/'wcbr_3dlrf_inference_efficiency_260909.md'
report.write_text('\n'.join(lines)+'\n')
print(report)
print(json.dumps(summary,indent=2))
