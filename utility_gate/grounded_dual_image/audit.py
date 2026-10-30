"""Audits and reports use only this run's namespace for writes."""
from __future__ import annotations
import csv,json,platform,subprocess
from pathlib import Path
from collections import Counter
from .common import ROOT,DATA,META,REPO,OFFICIAL,MIRROR,read_jsonl,write_json,sha256,status
from .stats import aggregates,paired

def initial_reports():
    meta=json.loads((ROOT/'audit/music_metadata.json').read_text())
    fixtures=json.loads((ROOT/'audit/legacy_gpu_regressions.json').read_text())
    native=json.loads((ROOT/'audit/native_interface_probe.json').read_text())
    media=read_jsonl(ROOT/'audit/media_music_dev_pilot.jsonl')
    write_json(ROOT/'audit/environment.json',{'python':platform.python_version(),'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'git_status':subprocess.check_output(['git','status','--short'],cwd=REPO,text=True),'old_fixtures':fixtures})
    (ROOT/'reports/M0_repo_audit.md').write_text(f'''STATUS: completed
WHY: 既有模型、数据路径、旧结果保护和原生接口已在当前 GPU 环境实测。
ACTUAL_VIDEO_SOURCES: {len(media)} 个 MUSIC 开发视频；另有历史 4383 个来源排重台账。
ACTUAL_QA_PAIRS: 119（首批开发）。
ACTUAL_AUDIO_VALID: 媒体音轨 {sum(r['status']=='ok' for r in media)}/{len(media)}；逐帧因果可用性另存 QA 行。
TOKEN_COUNTS: 4 帧 Stage1/Stage3 为 256；Full Vision 为 785（784 内容 + 1 换行）；native image 实测每图 {native['1']['native_inserted_tokens']}，两图 {native['2']['native_inserted_tokens']}。
PROMPT_VERSION: qwen_1_5 原聊天模板，MUSIC 改为公开的短答案请求；不虚构 ABCD。
GT_ORACLE_INPUT: false
MODEL_SHA256: ../sha_locks.json；旧 4805 个文件锁在 audit/protected_assets.json。
DATA_SHA256: audit/media_music_dev_pilot.jsonl
TEST_SELECTION_LOCK: 初始角色 manifests 已固定；任何确认预测前另立 pretest_lock.json。
COMMANDS_AND_LOGS: commands_executed.jsonl、logs/M0_gpu_regressions.log、logs/M1_pilot.log。
OBSERVED_NUMBERS: 原 Stage1 P/Z/生成 [34,151645]、原 Stage3 P/Z/生成 [32,151645] 均逐值一致。模型全部冻结。
LIMITATIONS: native 多图已测图像输入契约，真实双图 QA 重跑 fixture 在 M3 完成。已有 Stage1/Stage3 训练历史不同，不是只有音频不同的同权重消融。

视觉路径为 SigLIP contextual 27×27×1152 → 原 mm_projector 27×27×3584 → video 2D bilinear pool 14×14 → P[64,196] → Z=P@V → 冻结 Qwen2。Stage3 在 query 侧使用 BEATs 与时间编码；视觉值保持原值。extension.enabled=false 是旧默认路径，与选择 Stage1 always_off 并不等价。本轮未改旧路由器或 wrapper。
''')
    (ROOT/'reports/M0_dataset_research.md').write_text(f'''STATUS: partial
WHY: MUSIC 公开逐视频镜像和官方更新标注可用；原始 YouTube ID/offset 未在所审计官方元数据发布，独立性只能核验到真实视频 ID、文件和已知 hash。
ACTUAL_VIDEO_SOURCES: 开发计划 200、确认计划 500；先获取 32 开发媒体，不盲目获取全 48GB。
ACTUAL_QA_PAIRS: 开发 715、确认 701，均保留选中视频对应原始 split 的全部问题。
ACTUAL_AUDIO_VALID: 32/32 首批媒体含音轨；后续媒体和逐帧状态存 inventory 与 QA 明细。
TOKEN_COUNTS: 同 M0_repo_audit.md。
PROMPT_VERSION: S0（先跑 QA 基线）。
GT_ORACLE_INPUT: false
MODEL_SHA256: ../sha_locks.json
DATA_SHA256: audit/music_metadata.json、media_*.jsonl。
TEST_SELECTION_LOCK: 先保留 500 个 official test 真实视频 ID，再从 train 排除这 500 个，选 200 个 dev；seed1234 SHA 排序，与模型结果无关。
COMMANDS_AND_LOGS: logs/acquire_*.log，commands_executed.jsonl。
OBSERVED_NUMBERS: 官方 updated train/val/test={meta['native_splits']['train']['qa']}/{meta['native_splits']['val']['qa']}/{meta['native_splits']['test']['qa']}；镜像三份与官方 json_update 完全相等。原始官方早期 json 存在删除题和改答案，未混用。
LIMITATIONS: 官方分割以 QA 为单位，train/test 的真实视频 ID 交集 {meta['native_splits']['train_test_video_overlap']}，不能直接声称其来源独立。所有合成视频排除，避免未知 donor 交叉。新集合与历史已知 YouTube IDs 的语义映射无法核验，结论必须保留这个限制。

官方代码/标注 commit `{OFFICIAL}`：[MUSIC-AVQA](https://github.com/GeWu-Lab/MUSIC-AVQA)。其代码 MIT 不代表原始 YouTube 媒体获得再分发许可。镜像 UnFaZeD07/Music-AVQA revision `{MIRROR}`，匿名公开逐文件可读；只在本机研究，不再发布媒体。首次普通 git clone 传输停在 32MB 且没有 checkout，为遵守先取标注的要求已终止；保留不完整目录，之后从官方固定 commit raw URL 获取源码和标注。未绕过登录或访问控制。

AVQA-Hard 1696 索引全部连接到官方 AVQA val，1633 题/1630 已知未检视原始 YouTube 来源，但本地尚无其媒体；需核验镜像时长与原任务片段是否一致。Valor32k test 26088 题、3464 来源及对应媒体文件；3438 来源不在已知历史台账。仅元数据统计不代表这些模型结果已经检视。Valor 为自动生成标签与 research-only 媒体条件；不使用 caption/source_tags/答案生成提示图。
''')
    (ROOT/'dataset_link_validation.md').write_text(f'''# Dataset link validation

- 官方源码/updated JSON：固定 commit {OFFICIAL}，公开 HTTP 200，SHA 见 audit/music_metadata.json。
- MUSIC 镜像：[UnFaZeD07/Music-AVQA](https://huggingface.co/datasets/UnFaZeD07/Music-AVQA/tree/{MIRROR})，公开逐 MP4 下载，获取记录含完整 URL、revision、bytes、SHA、实际 PyAV 媒体流。
- [AVQA-Hard](https://huggingface.co/datasets/gwkrsrch/avqa_hard)，revision edc5cdf46f6d498cba251e0721bf91a21dfb5049；只有索引，不能把 GPT-4o 输出当 Gold。原始 AVQA 内容仍受其原条款约束。
- [Valor32k](https://huggingface.co/datasets/inesriahi/valor32k-avqa-v2)，revision 1b916bd15c19302f319ee6dd1206c7555bb19d76；test 元数据和逐视频可读，声明 Research use only，标签质量需分层。
- 第一次探测系统 ffprobe 不存在，保留失败日志，改用当前环境已有 PyAV；未改共享依赖环境。GitHub API 403 rate-limit 保留记录，改读同一官方公开 raw 文件。
''')
    status('M0','partial',reason='repo/interface audit complete; official original-source mapping absent; secondary media alignment audit pending')
    baseline_report('music_dev_pilot')

def baseline_report(role):
    rows=[]
    for path in sorted((ROOT/'qa_eval').glob(f'baseline_{role}_shard*.jsonl')):rows+=read_jsonl(path)
    out=aggregates(rows);write_json(ROOT/f'qa_eval/baseline_aggregate_{role}.json',out)
    by={s:[r for r in rows if r['strategy']==s] for s in out}
    pairs={s:paired(by[s],by['V64']) for s in by if s!='V64'}
    write_json(ROOT/f'qa_eval/paired_endpoint_matrix_{role}.json',pairs)
    table='\n'.join(f"| {s} | {v['all']['correct']}/{v['all']['accuracy']['n_questions']} | {v['all']['accuracy']['mean']:.4f} | {v['all']['custom_gold_string_NLL']['mean']:.4f} |" for s,v in out.items())
    (ROOT/f'reports/M1_{role}_baselines.md').write_text(f'''STATUS: completed
WHY: 同一真实媒体/题目/4 帧观察/生成与评分，冻结四路径逐题运行。
ACTUAL_VIDEO_SOURCES: {len({r['source_id'] for r in rows})}
ACTUAL_QA_PAIRS: {len(by['V64'])}
ACTUAL_AUDIO_VALID: {sum(all(r['audio_valid']) for r in by['V64'])} 个视频问题的全部4个音频窗口有效；逐帧理由在原始表。
TOKEN_COUNTS: V64/A64/A64-null=256，V196=785。
PROMPT_VERSION: S0, qwen_1_5; greedy max_new_tokens=16。
GT_ORACLE_INPUT: false
MODEL_SHA256: ../sha_locks.json
DATA_SHA256: manifests/{role}_available.jsonl 与获取台账。
TEST_SELECTION_LOCK: 开发集；未使用确认结果。
COMMANDS_AND_LOGS: logs/M1_*.log，qa_eval/baseline_{role}_shard*.jsonl。
OBSERVED_NUMBERS:

| Path | correct/QAs | normalized EM | custom Gold string NLL |
|---|---:|---:|---:|
{table}

LIMITATIONS: 这是官方更新答案上的冻结 VLM 开放生成 EM，不是复现官方训练分类器分数。主 EM 显式统一大小写/空白/下划线/末尾标点与整答案数字0..11；严格无数字别名 EM 同时保存。Gold NLL 是官方 canonical 字符串的全词表 teacher-forced token NLL 总和，不含 EOS，不是官方42类分类 NLL；不得把它与 MCQA NLL 混合。CI按视频ID聚类，不把题目当独立来源。完整观察 QA 均匀4帧，逐帧音频严格[t-2,t]；未做稠密视频复现。
''')
    return out

if __name__=='__main__':initial_reports()
