import json
import re
from typing import Any, Dict, Union, Optional

JsonLike = Union[str, Dict[str, Any]]

# ============================================================================
# 工具函数（参考model_comparison_simple.py）
# ============================================================================
def normalize_time_string(time_str: str) -> str:
    """标准化时间字符串格式，移除时区信息，统一格式"""
    if not isinstance(time_str, str):
        return str(time_str)

    # 移除所有空格
    time_str = time_str.strip()

    # 移除时区信息（Z, +XX:XX, -XX:XX等）
    time_str = re.sub(r'Z|[+-]\d{2}:\d{2}|[+-]\d{4}', '', time_str)

    # 处理各种时间格式
    # 1. "2023-12-01T10:00-00" -> "2023-12-01T10:00:00"
    # 2. "2023-12-01T10:00:00" -> "2023-12-01T10:00:00"
    # 3. "10:00:00" -> "10:00:00"
    # 4. "10:00" -> "10:00:00"

    # 如果是完整的时间戳格式
    if 'T' in time_str:
        date_part, time_part = time_str.split('T', 1)
        # 处理时间部分
        time_parts = time_part.replace('-', ':').split(':')
        if len(time_parts) == 2:
            time_parts.append('00')  # 添加秒
        time_normalized = ':'.join(time_parts[:3])
        return f"{date_part}T{time_normalized}"
    else:
        # 只是时间部分
        time_parts = time_str.replace('-', ':').split(':')
        if len(time_parts) == 2:
            time_parts.append('00')  # 添加秒
        return ':'.join(time_parts[:3])


def normalize_time_value(time_val) -> Any:
    """标准化时间值，支持字符串和列表"""
    if isinstance(time_val, list):
        # 列表中的每个元素都标准化
        return [normalize_time_string(str(item)) for item in time_val]
    elif isinstance(time_val, str):
        return normalize_time_string(time_val)
    else:
        return time_val


def compare_time_values(val1, val2) -> bool:
    """比较两个时间值是否匹配（支持不同格式）"""
    # 标准化两个值
    norm_val1 = normalize_time_value(val1)
    norm_val2 = normalize_time_value(val2)

    # 直接比较标准化后的值
    return norm_val1 == norm_val2

def _filter_think_tags(text: str) -> str:
	"""过滤掉 <think> 标签及其内容，保留标签外的内容"""
	if not isinstance(text, str):
		return text

	# 处理成对的<think>...</think>标签
	pattern_paired = r'<think>.*?</think>'
	filtered_text = re.sub(pattern_paired, '', text, flags=re.DOTALL)

	# 处理只有一个<think>开头的情况（后面没有</think>）
	# 匹配<think>后面跟着换行符，然后是JSON内容
	pattern_single = r'<think>\s*\n'
	filtered_text = re.sub(pattern_single, '', filtered_text)

	# 移除末尾的特殊token（如<|endoftext|>）
	filtered_text = re.sub(r'<\|endoftext\|>$', '', filtered_text).strip()

	return filtered_text

def _safe_parse_json(obj: JsonLike) -> Optional[Dict[str, Any]]:
	"""安全解析JSON"""
	if isinstance(obj, dict):
		return obj
	if isinstance(obj, str):
		try:
			filtered_str = _filter_think_tags(obj)
			parsed = json.loads(filtered_str)
			# 确保返回的是字典
			if isinstance(parsed, dict):
				return parsed
			else:
				return None
		except Exception:
			return None
	return None

def _extract_fields(answer: JsonLike) -> Dict[str, Any]:
	"""提取关键字段"""
	data = _safe_parse_json(answer)
	if data is None:
		return {
			"source_ishighloadcell": None,
			"target_subnet_id": None,
			"target_me_id": None,
			"target_ldn": None,
			"highload_time": None,
			"result": None,
			"kpi": None,
		}

	target_raw = data.get("target")
	target = target_raw if isinstance(target_raw, dict) else {}
	load_unbalance_result = data.get("load_unbalance_result", {}) if isinstance(data.get("load_unbalance_result"), dict) else {}

	return {
		"source_ishighloadcell": data.get("source_ishighloadcell"),
		"target_subnet_id": target.get("subnet_id"),
		"target_me_id": target.get("me_id"),
		"target_ldn": target.get("ldn"),
		"highload_time": data.get("highload_time"),
		"result": load_unbalance_result.get("result"),
		"reason_zh": load_unbalance_result.get("reason_zh"),
		"reason_en": load_unbalance_result.get("reason_en"),
		"kpi": load_unbalance_result.get("kpi"),
	}

def _score_match(pred: Any, ref: Any, field_name: str = "") -> float:
	"""灵活匹配：支持不同字段类型的匹配逻辑"""
	if pred is None and ref is None:
		return 1.0
	if pred is None or ref is None:
		return 0.0

	# 特殊处理时间字段
	if field_name == "highload_time":
		return 1.0 if compare_time_values(pred, ref) else 0.0

	# 默认严格匹配
	return 1.0 if pred == ref else 0.0

def compute_reward(answer: JsonLike, reference: JsonLike) -> Dict[str, Any]:
	"""计算奖励分数 - 参考model_comparison_simple.py的匹配逻辑
	- 基于字段级别的匹配，支持更灵活的比较
	- 最终得分基于各字段的匹配情况
	"""
	pred_data = _safe_parse_json(answer)
	ref_data = _safe_parse_json(reference)

	# 确保pred_data和ref_data是字典类型
	if not isinstance(pred_data, dict):
		pred_data = None
	if not isinstance(ref_data, dict):
		ref_data = None

	if pred_data is None or ref_data is None:
		return {
			"total": 0.0,
			"components": {
				"source_ishighloadcell": 0.0,
				"target_subnet_id": 0.0,
				"target_me_id": 0.0,
				"target_ldn": 0.0,
				"highload_time": 0.0,
				"result": 0.0,
				"reason_zh": 0.0,
				"reason_en": 0.0,
				"kpi": 0.0,
			},
			"parsed_pred": pred_data,
			"parsed_ref": ref_data,
		}

	def _is_empty_value(v: Any) -> bool:
		return v is None or v == "" or v == [] or v == {}

	# 初始化各字段分数
	field_scores = {
		"source_ishighloadcell": 0.0,
		"target_subnet_id": 0.0,
		"target_me_id": 0.0,
		"target_ldn": 0.0,
		"highload_time": 0.0,
		"result": 0.0,
		"reason_zh": 0.0,
		"reason_en": 0.0,
		"kpi": 0.0,
	}

	# 1. 比较 source_ishighloadcell
	if "source_ishighloadcell" in pred_data and "source_ishighloadcell" in ref_data:
		field_scores["source_ishighloadcell"] = _score_match(
			pred_data["source_ishighloadcell"],
			ref_data["source_ishighloadcell"],
			"source_ishighloadcell"
		)

	# 2. 比较 highload_time（支持时间格式灵活匹配）
	if "highload_time" in pred_data and "highload_time" in ref_data:
		field_scores["highload_time"] = _score_match(
			pred_data["highload_time"],
			ref_data["highload_time"],
			"highload_time"
		)

	# 3. 比较 target 子字段（参考model_comparison_simple.py的逻辑）
	if "target" in pred_data and "target" in ref_data:
		pred_target = pred_data["target"]
		ref_target = ref_data["target"]

		if isinstance(pred_target, dict) and isinstance(ref_target, dict):
			# 检查空对象情况
			pred_empty = len(pred_target) == 0
			ref_empty = len(ref_target) == 0

			if pred_empty and ref_empty:
				# 两个都是空对象，所有target子字段都匹配
				field_scores["target_subnet_id"] = 1.0
				field_scores["target_me_id"] = 1.0
				field_scores["target_ldn"] = 1.0
			elif pred_empty != ref_empty:
				# 一个空一个不空，都不匹配
				field_scores["target_subnet_id"] = 0.0
				field_scores["target_me_id"] = 0.0
				field_scores["target_ldn"] = 0.0
			else:
				# 两个都不是空对象，比较子字段
				for sub_field in ["subnet_id", "me_id", "ldn"]:
					field_key = f"target_{sub_field}"
					if sub_field in ref_target:
						field_scores[field_key] = _score_match(
							pred_target.get(sub_field),
							ref_target.get(sub_field),
							field_key
						)
					elif sub_field not in pred_target:
						field_scores[field_key] = 1.0  # 两者都没有该字段
					else:
						field_scores[field_key] = 0.0  # pred有但ref没有

	# 4. 比较 load_unbalance_result 子字段
	if "load_unbalance_result" in pred_data and "load_unbalance_result" in ref_data:
		pred_result = pred_data["load_unbalance_result"]
		ref_result = ref_data["load_unbalance_result"]

		if isinstance(pred_result, dict) and isinstance(ref_result, dict):
			for sub_field in ["result", "reason_zh", "reason_en", "kpi"]:
				if sub_field in ref_result:
					field_scores[sub_field] = _score_match(
						pred_result.get(sub_field),
						ref_result.get(sub_field),
						sub_field
					)
				elif sub_field not in pred_result:
					field_scores[sub_field] = 1.0  # 两者都没有该字段
				else:
					field_scores[sub_field] = 0.0  # pred有但ref没有

	# 计算总分（9个字段的平均值）
	total = sum(field_scores.values()) / 9.0

	return {
		"total": float(total),
		"components": field_scores,
		"parsed_pred": pred_data,
		"parsed_ref": ref_data,
	}

def rule_reward(data_source, extra_info, solution_str, ground_truth, method="strict", format_score=0.0, score=1.0, config=None) -> float:
	"""自定义VERL入口函数"""
	res = compute_reward(solution_str, ground_truth)
	return float(res["total"])