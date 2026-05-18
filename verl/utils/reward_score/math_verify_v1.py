#!/usr/bin/env python3
"""
最终修正和优化的数学答案验证奖励函数
- 修正了 extract_last_boxed 函数的括号匹配逻辑
- 修正了正则表达式转义问题
- 优化了测试用例和调试信息
"""

import re
import logging
from typing import Optional, Dict, Any, Tuple

# --- 全局设置与常量 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 尝试导入math_verify库
try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    MATH_VERIFY_AVAILABLE = True
    logging.info("Math-verify库已成功加载。")
except ImportError:
    logging.critical("CRITICAL: Math-verify库导入失败！")
    MATH_VERIFY_AVAILABLE = False
    TimeoutException = Exception

# --- 正则表达式预编译 ---
TEX_COMMAND_PATTERN = re.compile(r"\\(text|textbf|overline)\{([^}]*)\}")
BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")
MINERVA_ANSWER_PATTERN = re.compile(r"(?i)Answer\s*:\s*([^\n]+)")

SUBSTITUTIONS = [
    ("an ", ""), (".$", "$"), ("\\$", ""), (r"\ ", ""), (" ", ""),
    ("mbox", "text"), (",\\text{and}", ","), ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS_LIST = [
    "square", "ways", "integers", "dollars", "mph", "inches", "hours", "km",
    "units", "\\ldots", "sue", "points", "feet", "minutes", "digits", "cents",
    "degrees", "cm", "gm", "pounds", "meters", "meals", "edges", "students",
    "childrentickets", "multiples", "\\text{s}", "\\text{.}", "\\text{\ns}",
    "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}", r"\mathrm{th}",
    r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]
REMOVED_EXPRESSIONS_PATTERN = re.compile("|".join(map(re.escape, REMOVED_EXPRESSIONS_LIST)))


# --- 核心函数 ---
def validate_hybird_format(response: str, prompt: str) -> float:
    """
    Validate the usage of <think>...</think> tags in the response based on the prompt instructions.

    Rules:
    - If prompt ends with "\\think" or has no label, expect exactly one <think> section with non-empty content.
    - If prompt ends with "\\no_think", expect exactly one <think> section with empty content (exactly "\n\n").
    
    Returns True if the response matches the expected format, False otherwise.
    """
    prompt_text = prompt.strip()

    expects_think = False
    expects_no_think = False

    if prompt_text.endswith(r"no_think"):
        expects_no_think = True
    elif prompt_text.endswith(r"think"):
        expects_think = True
    else:
        # Default: assume thinking is expected if no explicit label
        expects_think = True

    # Use regex to find all <think>...</think> blocks (non-greedy, DOTALL to match newlines)
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    think_sections = pattern.findall(response)

    # Must have exactly one <think> block
    if len(think_sections) != 1:
        return 0.0

    content = think_sections[0]

    if expects_think:
        # Content must be non-empty when stripped (not just whitespace or newlines)
        return 0.1 if content.strip() != "" else 0.0

    elif expects_no_think:
        # Content must be exactly two newlines: "\n\n"
        return 0.1 if content == "\n\n" else 0.0

    return 0.0

def extract_last_boxed(text: str) -> Optional[str]:
    """
    精确提取字符串中最后一个 \\boxed{...} 表达式。
    修正了括号匹配逻辑，确保正确处理嵌套括号。
    """
    if not text:
        return None
    
    # 查找最后一个 \\boxed{ 的位置
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    
    # 从 \\boxed{ 开始，找到匹配的右括号
    i = idx + 7  # 跳过 "\\boxed{"
    brace_count = 1
    result_content = ""
    
    while i < len(text) and brace_count > 0:
        char = text[i]
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
        
        if brace_count > 0:  # 只有当还有未匹配的括号时才添加字符
            result_content += char
        i += 1
    
    if brace_count == 0:
        return f"\\boxed{{{result_content}}}"
    else:
        return None

def remove_boxed(s: str) -> str:
    """移除LaTeX boxed命令的包装。"""
    if not s or not s.startswith("\\boxed{") or not s.endswith("}"):
        return s
    return s[len("\\boxed{"):-1]

def is_correct_strict_box(pred_text: str, gt_text: str) -> Tuple[bool, Optional[str]]:
    """严格的boxed答案验证。"""
    boxed_pred = extract_last_boxed(pred_text)
    if boxed_pred is None:
        return False, None
    
    extracted_pred = remove_boxed(boxed_pred)
    logging.debug(f"严格验证 - 提取: '{extracted_pred}', 标准: '{gt_text}'")
    return extracted_pred == gt_text, extracted_pred

def verify_with_math_verify(pred_full_boxed: str, gt_text: str, timeout_score: float = 0.0) -> float:
    """
    使用math_verify进行语义验证，输入必须是对称的boxed格式。
    """
    if not MATH_VERIFY_AVAILABLE or not pred_full_boxed or not gt_text:
        return 0.0
    
    try:
        verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(LatexExtractionConfig(),)
        )
        
        # 确保标准答案格式正确
        if not gt_text.startswith("\\boxed{"):
            ground_truth_boxed = f"\\boxed{{{gt_text}}}"
        else:
            ground_truth_boxed = gt_text
        
        logging.debug(f"Math-verify输入 - 预测: '{pred_full_boxed}', 标准: '{ground_truth_boxed}'")
        
        ret_score, details = verify_func([ground_truth_boxed], [pred_full_boxed])
        
        logging.debug(f"Math-verify结果: {ret_score}, 详情: {details}")
        return float(ret_score)
        
    except Exception as e:
        logging.warning(f"Math-verify验证时异常: {e}")
        return 0.0
def compute_score(
    model_output: str,
    ground_truth: str,
    verification_strategy: str = "combined",
    timeout_score: float = 0.0,
    math_verify_threshold: float = 0.9
) -> float:  # -> 将返回类型注解修改为 float
    """
    计算单个样本的分数，并确保总是返回一个浮点数。
    """
    if not model_output or not ground_truth:
        return 0.0
    prompt = "give me a answer!!"
    format_rd = validate_hybird_format(model_output, prompt)
    model_output = model_output[-800:]
    # print(f"model_output: {model_output}")
    # print(f"ground_truth: {ground_truth}")
    try:
        # 这里保留你所有的 if/elif 验证逻辑...
        if verification_strategy == "strict_box":
            is_correct, _ = is_correct_strict_box(model_output, ground_truth)
            return 1.0 if is_correct else 0.0
        
        elif verification_strategy == "combined":
            is_strict_correct, _ = is_correct_strict_box(model_output, ground_truth)
            if is_strict_correct:
                return 1.0
            else:
                pred_full_boxed = extract_last_boxed(model_output)
                if pred_full_boxed and MATH_VERIFY_AVAILABLE:
                    score = verify_with_math_verify(pred_full_boxed, ground_truth, timeout_score)
                    return score if score > math_verify_threshold else 0.0
                else:
                    return 0.0+format_rd
        else:
            raise ValueError(f"未知的验证策略: {verification_strategy}")

    except Exception as e:
        logging.error(f"为输出 '{model_output[:50]}...' 计算分数时发生错误: {e}", exc_info=False)
        return 0.0+format_rd  # 即使发生异常，也返回一个代表惩罚的浮点数 0.0

def test_reward_function():
    """
    更全面的测试套件（已修正字符串转义）
    """
    print("\n" + "="*20 + " 测试修正后的奖励函数 " + "="*20)
    
    # 关键修正：使用原始字符串避免转义问题
    test_cases = [
        {
            "desc": "语义正确, 格式不同", 
            "output": r"结果为 \boxed{0.5}", 
            "gt": r"\frac{1}{2}", 
            "strategy": "combined", 
            "expected": True if MATH_VERIFY_AVAILABLE else False
        },
        {
            "desc": "语义正确, 格式相同", 
            "output": r"所以答案是 \boxed{\frac{1}{2}}", 
            "gt": r"\frac{1}{2}", 
            "strategy": "combined", 
            "expected": True
        },
        {
            "desc": "组合策略 - 语义正确", 
            "output": r"结果为 \boxed{1/2}", 
            "gt": r"0.5", 
            "strategy": "combined", 
            "expected": True if MATH_VERIFY_AVAILABLE else False
        },
        {
            "desc": "严格验证失败，语义也失败", 
            "output": r"答案是 \boxed{1/3}", 
            "gt": r"0.5", 
            "strategy": "combined", 
            "expected": False
        },
        {
            "desc": "严格验证正确", 
            "output": r"The answer is just \boxed{100}", 
            "gt": r"100", 
            "strategy": "combined", 
            "expected": True
        },
        {
            "desc": "测试复杂LaTeX", 
            "output": r"答案是 \boxed{\frac{a+b}{c}}", 
            "gt": r"\frac{a+b}{c}", 
            "strategy": "combined", 
            "expected": True
        },
    ]
    
    # 设置调试级别以查看详细信息
    logging.getLogger().setLevel(logging.DEBUG)
    
    for i, test in enumerate(test_cases):
        if not MATH_VERIFY_AVAILABLE and test['expected'] and "语义" in test['desc']:
            print(f"\n--- 跳过测试用例 {i+1}: {test['desc']} (Math-verify不可用) ---")
            continue
            
        strategy = test.get("strategy", "combined")
        print(f"\n--- 测试用例 {i+1}: {test['desc']} (策略: {strategy}) ---")
        print(f"  模型输出: {test['output']}")
        print(f"  标准答案: {test['gt']}")
        
        results = compute_score(test['output'], test['gt'], verification_strategy=strategy)
        
        print(f"  提取答案: {results.get('extracted_answer')}")
        print(f"  严格Box正确: {results.get('strict_box_correct', 'N/A')}")
        print(f"  Math-verify分数: {results.get('math_verify_score', 0.0):.3f}")
        print(f"  最终分数: {results['score']:.3f}")
        print(f"  最终正确: {results['final_correct']}")
        
        if results.get('error'):
            print(f"  错误: {results['error']}")
        
        success = results['final_correct'] == test['expected']
        print(f"  测试结果: {'✓ 通过' if success else '✗ 失败'}")
        
        if not success:
            print(f"    期望: {test['expected']}, 实际: {results['final_correct']}")


def debug_extract_function():
    """调试extract_last_boxed函数"""
    print("\n" + "="*20 + " 调试extract_last_boxed函数 " + "="*20)
    
    test_strings = [
        r"结果为 \boxed{0.5}",
        r"所以答案是 \boxed{\frac{1}{2}}",
        r"答案是 \boxed{100}",
        r"答案是 \boxed{\frac{a+b}{c}}",
        r"没有boxed格式的文本",
        r"错误的格式 \boxed{未闭合",
    ]
    
    for i, test_str in enumerate(test_strings, 1):
        result = extract_last_boxed(test_str)
        print(f"测试 {i}: '{test_str}' -> '{result}'")


if __name__ == "__main__":
    # 先调试extract函数
    debug_extract_function()
    
    # 然后运行完整测试
    test_reward_function()