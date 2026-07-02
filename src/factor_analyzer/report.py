"""Markdown / JSON 分析报告"""



from __future__ import annotations



import json

from datetime import datetime

from pathlib import Path

from typing import Any

from zoneinfo import ZoneInfo





def build_report_payload(

    *,

    window_info: dict[str, Any],

    ic_summaries: dict[str, dict[str, Any]],

    quintile_by_score: dict[str, dict[str, Any]],

    verdict: dict[str, Any],

    component_ic: dict[str, float],

    factor_config_path: str,

    return_column: str,

    return_valid_count: int,

    panel_rows: int,

    top3_stats: dict[str, Any] | None = None,

    top1_stats: dict[str, Any] | None = None,

    walk_forward: dict[str, Any] | None = None,

    optimization: dict[str, Any] | None = None,

) -> dict[str, Any]:

    """组装报告 JSON 结构"""

    return {

        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),

        "factor_config": factor_config_path,

        "return_column": return_column,

        "return_storage": "output/archive/stock_pool/",

        "window": window_info,

        "return_valid_count": return_valid_count,

        "panel_rows": panel_rows,

        "primary_score": window_info.get("primary_score", "final_score"),

        "ic_summaries": ic_summaries,

        "quintile": quintile_by_score,

        "component_ic": component_ic,

        "verdict": verdict,

        "top3_stats": top3_stats or {},

        "top1_stats": top1_stats or {},

        "walk_forward": walk_forward or {},

        "optimization": optimization or {},

    }





def render_markdown_report(payload: dict[str, Any]) -> str:

    """生成 Markdown 报告正文"""

    window = payload.get("window", {})

    verdict = payload.get("verdict", {})

    primary = payload.get("primary_score", "final_score")

    ic_primary = payload.get("ic_summaries", {}).get(primary, {})

    quintile = payload.get("quintile", {}).get(primary, {})

    return_col = payload.get("return_column", "future_return_20")



    lines = [

        "# 因子有效性分析报告",

        "",

        f"- 生成时间: {payload.get('generated_at')}",

        f"- 因子配置: `{payload.get('factor_config')}`",

        f"- 收益列: `{return_col}`（位于 `{payload.get('return_storage')}`）",

        f"- 分析窗口: {len(window.get('analysis_dates', []))} 个交易日 "

        f"（可算收益 {len(window.get('return_ready_dates', []))} 日）",

        f"- 收益截止日: {window.get('return_end')}",

        f"- 收益 horizon: {window.get('return_horizon')} 个交易日",

        f"- 面板样本: {payload.get('panel_rows')} 行",

        "",

        "## 结论",

        "",

    ]



    status = verdict.get("status")

    if status == "insufficient_sample":

        lines.append(

            f"**样本不足**：当前仅 {len(window.get('analysis_dates', []))} 个可分析交易日，"

            f"低于最低要求 {window.get('requested_min', 60)} 日。"

            "以下为参考指标，不做正式失效判定。"

        )

    else:

        status_label = verdict.get("status_label", "未知")

        lines.append(f"**当前打分模型（基于 {return_col}）: {status_label}**")

        lines.append(

            f"- 规则通过: {verdict.get('passed_count')}/{len(verdict.get('rules', []))} "

            f"（至少 {verdict.get('pass_min_rules')} 项通过）"

        )



    lines.extend(["", "## 核心指标（final_score）", ""])

    lines.append("| 指标 | 值 |")

    lines.append("|------|-----|")

    lines.append(f"| IC 均值 | {ic_primary.get('ic_mean')} |")

    lines.append(f"| IC 标准差 | {ic_primary.get('ic_std')} |")

    lines.append(f"| IC_IR | {ic_primary.get('ic_ir')} |")

    lines.append(f"| IC 胜率 | {ic_primary.get('ic_positive_ratio')} |")

    lines.append(f"| IC 有效日数 | {ic_primary.get('ic_days')} |")

    lines.append(f"| 五分位价差 (Q5-Q1) | {quintile.get('quintile_spread')}% |")

    lines.append(f"| 五分位单调 | {quintile.get('monotonic')} |")

    lines.append(f"| Top20% 超额 | {quintile.get('top20_excess')}% |")



    top1 = payload.get("top1_stats") or {}

    top3 = payload.get("top3_stats") or {}

    if top1.get("pick_count", 0) > 0 or top3.get("pick_count", 0) > 0:

        lines.extend(

            [

                "",

                "## 操盘参考（不参与失效判定）",

                "",

                f"模拟每日 `{primary}` **第 1 名**与**前 3 只**，"

                f"以 `{return_col} > 0` 视为单笔获胜（到期盈利；"

                "未模拟 ±10% 止盈止损路径）。",

                "",

            ]

        )

        if top1.get("pick_count", 0) > 0:

            wr1 = top1.get("win_rate") or 0

            lines.append(

                f"- Top1 单笔胜率: **{wr1 * 100:.2f}%**"

                f"（{top1.get('win_count')}/{top1.get('pick_count')} 笔）"

            )

        if top3.get("pick_count", 0) > 0:

            tk = top3.get("top_k", 3)

            wr = top3.get("win_rate") or 0

            dwr = top3.get("daily_all_win_rate") or 0

            if top1.get("pick_count", 0) > 0:

                lines.append("")

            lines.extend(

                [

                    f"- Top{tk} 单笔胜率: **{wr * 100:.2f}%**"

                    f"（{top3.get('win_count')}/{top3.get('pick_count')} 笔）",

                    f"- Top{tk} 平均 {return_col}: {top3.get('avg_return_pct')}%",

                    f"- Top{tk} 二十日最大跌幅: **{top3.get('max_loss_pct')}%**"
                    f"（全部 {top3.get('pick_count')} 笔中单笔最差）",

                    f"- Top{tk} 每日最差一只平均 {return_col}: {top3.get('avg_day_worst_return_pct')}%",

                    f"- 信号日数: {top3.get('signal_days')} 日",

                    f"- 当日 Top{tk} 全部为正的比例: {dwr * 100:.2f}%",

                ]

            )



    wf = payload.get("walk_forward") or {}

    if wf.get("enabled"):

        lines.extend(_render_walk_forward_section(wf, primary, return_col))



    qmeans = quintile.get("quintile_means") or {}

    if qmeans:

        lines.extend(["", f"### 五分位平均 {return_col} (%)", ""])

        for q in sorted(qmeans.keys()):

            lines.append(f"- Q{q}: {qmeans[q]}%")



    rules = verdict.get("rules") or []

    if rules:

        lines.extend(["", "## 判定规则明细", ""])

        for r in rules:

            mark = "通过" if r.get("passed") else "未通过"

            lines.append(f"- [{mark}] {r.get('name')}: {r.get('detail')}")



    ic_all = payload.get("ic_summaries", {})

    if len(ic_all) > 1:

        lines.extend(["", "## 各类因子 IC 汇总", ""])

        lines.append("| 分数列 | IC均值 | IC_IR | IC胜率 |")

        lines.append("|--------|--------|-------|--------|")

        for col, s in ic_all.items():

            lines.append(

                f"| {col} | {s.get('ic_mean')} | {s.get('ic_ir')} | {s.get('ic_positive_ratio')} |"

            )



    comp_ic = payload.get("component_ic") or {}

    if comp_ic:

        lines.extend(["", "## 成分因子平均 IC", ""])

        for field, ic in sorted(comp_ic.items(), key=lambda x: x[1], reverse=True):

            lines.append(f"- `{field}`: {ic}")



    if status == "ineffective":

        lines.extend(

            [

                "",

                "## 后续操作",

                "",

                "模型判定失效，请查看 `output/analysis/proposed/factor_config.proposed.yaml`，",

                "人工审阅后替换 `config/factor_config.yaml`。",

            ]

        )

    else:

        opt = payload.get("optimization") or {}

        if opt.get("proposed_written"):

            rec = opt.get("recommend_replace")

            lines.extend(["", "## 后续操作", ""])

            if rec is True:

                lines.append(

                    "已生成 `factor_config.proposed.yaml`，**样本外测试集指标改善**，"

                    "建议审阅后替换 `config/factor_config.yaml`。"

                )

            else:

                lines.append(

                    "已生成 `factor_config.proposed.yaml`，请结合 Walk-forward 段落人工审阅。"

                )

        elif opt.get("skip_reason"):

            lines.extend(["", "## 后续操作", "", opt["skip_reason"] + "。"])



    return "\n".join(lines) + "\n"





def _render_walk_forward_section(

    wf: dict[str, Any],

    primary: str,

    return_col: str,

) -> list[str]:

    """Walk-forward 样本外评估 Markdown 段落"""

    split = wf.get("split", {})

    metric = wf.get("oos_primary_metric", "ic_ir")

    metric_label = {

        "ic_ir": "IC_IR",

        "ic_mean": "IC 均值",

        "quintile_spread": "五分位价差",

    }.get(metric, metric)

    opt_method = wf.get("optimization_method", "ridge_regime")
    method_label = {
        "ridge_regime": "Ridge 成分定权 + Regime 大类定权 + 因子池增删",
        "ic_heuristic": "IC_IR 启发式调权 + 因子池增删",
    }.get(opt_method, opt_method)
    meta = wf.get("proposed_meta") or {}
    ridge_alpha = meta.get("ridge_alpha")

    lines = [

        "",

        "## 因子优化 Walk-forward（Ridge + Regime）",

        "",

        f"优化方法: **{method_label}**"
        + (f"，Ridge α={ridge_alpha}" if ridge_alpha is not None else ""),

        "",

        f"训练 / 验证 / 测试切分（`{split.get('mode', 'fixed')}`）：",

        f"- 训练: {len(split.get('train_dates', []))} 日"

        f"（{ _date_range(split.get('train_dates')) }）",

        f"- 验证: {len(split.get('validate_dates', []))} 日"

        f"（{ _date_range(split.get('validate_dates')) }）",

        f"- 测试: {len(split.get('test_dates', []))} 日"

        f"（{ _date_range(split.get('test_dates')) }）",

        "",

        f"样本外主指标: `{primary}` 的 {metric_label}（Top-K 胜率不参与是否生成配置）",

        "",

        "| 分段 | 当前配置 | 建议配置 |",

        "|------|----------|----------|",

    ]

    for label, title in (

        ("train", "训练"),

        ("validate", "验证"),

        ("test", "测试(OOS)"),

    ):

        b = _metric_from_eval(wf.get("baseline", {}).get(label, {}), metric)

        p = _metric_from_eval(wf.get("proposed", {}).get(label, {}), metric)

        lines.append(f"| {title} | {b} | {p} |")

    imp = wf.get("test_improvement", {})

    lines.extend(

        [

            "",

            f"- 测试集 {metric_label}: 当前 **{imp.get('baseline')}** → 建议 **{imp.get('proposed')}**",

            f"- 样本外是否改善: **{'是' if imp.get('improved') else '否'}**",

            f"- 是否推荐替换: **{'是' if wf.get('recommend_replace') else '否'}**",

        ]

    )

    lines.extend(_render_factor_pool_section(wf.get("factor_pool") or {}))

    return lines





def _render_factor_pool_section(pool: dict[str, Any]) -> list[str]:
    """因子池增删建议（报告参考；是否写入 proposed 由样本外门禁决定）"""
    if not pool.get("enabled"):
        return []

    lines = ["", "### 因子池增删建议", ""]
    removals = pool.get("removal_candidates") or []
    additions = pool.get("addition_candidates") or []
    applied = pool.get("applied_changes") or []

    if removals:
        lines.append("**建议剔除（训练集 IC_IR 偏弱）:**")
        for r in removals:
            lines.append(
                f"- `{r.get('factor')}.{r.get('field')}` IC_IR={r.get('ic_ir')} "
                f"（{r.get('ic_days')} 日）"
            )
    else:
        lines.append("- 建议剔除: 无")

    if pool.get("addition_skipped_reason"):
        lines.append(f"- 建议新增: {pool['addition_skipped_reason']}")
    elif additions:
        lines.append("**建议新增（白名单候选）:**")
        for a in additions:
            lines.append(
                f"- `{a.get('factor')}.{a.get('field')}` IC_IR={a.get('ic_ir')} "
                f"（{a.get('ic_days')} 日）"
            )
    else:
        lines.append("- 建议新增: 无（或未达样本门槛）")

    if applied:
        lines.extend(["", "**已纳入本次 proposed 验证的变更:**"])
        for ch in applied:
            lines.append(f"- {ch}")
    else:
        lines.append("")
        lines.append("本次未对 proposed 做增删（无候选或仅报告参考）。")

    return lines


def _date_range(dates: list[str] | None) -> str:

    if not dates:

        return "-"

    if len(dates) == 1:

        return dates[0]

    return f"{dates[0]} … {dates[-1]}"





def _metric_from_eval(eval_result: dict[str, Any], metric: str) -> str:

    if metric == "quintile_spread":

        val = eval_result.get("quintile", {}).get("quintile_spread")

    elif metric == "ic_mean":

        val = eval_result.get("ic", {}).get("ic_mean")

    else:

        val = eval_result.get("ic", {}).get("ic_ir")

    return str(val) if val is not None else "-"





def write_reports(

    payload: dict[str, Any],

    *,

    reports_dir: Path,

    prefix: str,

) -> tuple[Path, Path]:

    """写入 JSON 与 Markdown 报告"""

    reports_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")

    json_path = reports_dir / f"{prefix}_{stamp}.json"

    md_path = reports_dir / f"{prefix}_{stamp}.md"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path.write_text(render_markdown_report(payload), encoding="utf-8")

    return md_path, json_path

