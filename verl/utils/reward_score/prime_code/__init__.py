# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import traceback
import ast
import re

from .utils import check_correctness as apps_check_correctness




def _is_valid_python_code(code: str) -> bool:
    """Check if the given string is valid Python code by attempting to parse it."""
    if not code or not code.strip():
        return False

    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
    except Exception:
        # For other parsing errors, also consider it invalid
                return False


def extract_code(completion: str) -> str:
    """Extract code from completion. Returns empty string if no valid code found."""
    pattern = r"```(?:python)?\s*(.*?)\s*```"
    matches = re.findall(pattern, completion, re.DOTALL)
    if matches:
        extracted_code = matches[0].strip()
        if _is_valid_python_code(extracted_code):
            return extracted_code

    # If no valid code block found, return empty string
    return ""


def is_validator_expression(output_str: str) -> bool:
    """Heuristic to detect if output is a validator expression."""
    if not isinstance(output_str, str):
        return False
    # Check for common logical operators
    ops = [" or ", " and ", " != ", " == ", "not ", " in ", ">=", "<=", ">", "<"]
    return any(op in output_str for op in ops)


def compute_score(completion, test_cases, continuous=False):
    """
    Unified scoring function supporting:
      - std: stdin/stdout based testing
      - assert: function assertion testing
      - validator: logical expression evaluation
    """
    # Extract code from completion
    solution = extract_code(completion)

    # If no valid code extracted, return 0
    if not solution:
        return 0.0

    try:
        # Parse test_cases if it's a string
        if isinstance(test_cases, str):
            test_cases = json.loads(test_cases)

        # Support assert type testing
        if isinstance(test_cases, dict) and test_cases.get("call_type") == "assert":
            return _validate_with_assert(solution, test_cases, continuous)

        # Support validator expression type
        if isinstance(test_cases, dict) and "outputs" in test_cases:
            outputs = test_cases.get("outputs", [])
            if any(is_validator_expression(out) for out in outputs):
                return _validate_with_expression(solution, test_cases, continuous)

        # Default: std input/output testing
        try:
            res, metadata = apps_check_correctness(
                in_outs=test_cases,
                generation=solution,
                timeout=5,
                debug=False
            )
            success = all(x is True for x in res)
            return 1.0 if success else 0.0
        except Exception:
            pass  # Fall through to per-case testing

        # Per-case testing for std type
        test_cases_list = []
        inputs = test_cases.get("inputs", [])
        outputs = test_cases.get("outputs", [])
        for i in range(len(inputs)):
            test_cases_list.append({
                "inputs": [inputs[i]],
                "outputs": [outputs[i]]
            })

        if continuous:
            res_list = []
            for test_case_id, test_case in enumerate(test_cases_list):
                if test_case_id >= 10:  # Only test first 10
                    break

                try:
                    res, metadata = apps_check_correctness(
                        in_outs=test_case,
                        generation=solution,
                        timeout=10,
                        debug=False
                    )
                    res_list.extend(res)
                except Exception:
                    res_list.append(False)

            success = sum(res_list) / max(len(res_list), 1) if res_list else 0.0
            return success

        # If not continuous, try single test
        if test_cases_list:
            try:
                res, metadata = apps_check_correctness(
                    in_outs=test_cases_list[0],
                    generation=solution,
                    timeout=10,
                    debug=False
                )
                success = all(x is True for x in res)
                return 1.0 if success else 0.0
            except Exception:
                pass

        # All tests failed
        return 0.0

    except Exception:
        return 0.0


def _validate_with_assert(solution: str, test_cases: dict, continuous: bool):
    """Validate code using assert statements."""
    try:
        # Create restricted globals to prevent network access
        exec_globals = {
            '__builtins__': {
                '__import__': __import__,
                'abs': abs, 'all': all, 'any': any, 'bin': bin, 'bool': bool, 'bytearray': bytearray,
                'bytes': bytes, 'callable': callable, 'chr': chr, 'classmethod': classmethod,
                'complex': complex, 'dict': dict, 'dir': dir, 'divmod': divmod, 'enumerate': enumerate,
                'filter': filter, 'float': float, 'format': format, 'frozenset': frozenset,
                'getattr': getattr, 'globals': globals, 'hasattr': hasattr, 'hash': hash,
                'help': help, 'hex': hex, 'id': id, 'int': int, 'isinstance': isinstance,
                'issubclass': issubclass, 'iter': iter, 'len': len, 'list': list, 'locals': locals,
                'map': map, 'max': max, 'memoryview': memoryview, 'min': min, 'next': next,
                'object': object, 'oct': oct, 'open': open, 'ord': ord, 'pow': pow, 'property': property,
                'range': range, 'repr': repr, 'reversed': reversed, 'round': round, 'set': set,
                'setattr': setattr, 'slice': slice, 'sorted': sorted, 'staticmethod': staticmethod,
                'str': str, 'sum': sum, 'super': super, 'tuple': tuple, 'type': type, 'vars': vars,
                'zip': zip, 'print': print, 'Exception': Exception, 'ValueError': ValueError,
                'TypeError': TypeError, 'IndexError': IndexError, 'KeyError': KeyError,
                'AttributeError': AttributeError, 'ImportError': ImportError, 'NameError': NameError,
                'math': __import__('math'), 'random': __import__('random'), 'collections': __import__('collections'),
                'itertools': __import__('itertools'), 'functools': __import__('functools'), 'operator': __import__('operator'),
                're': __import__('re'), 'json': __import__('json'), 'datetime': __import__('datetime'), 'time': __import__('time'),
                'typing': __import__('typing'),
            }
        }

        # Add common type annotations to globals
        import typing
        exec_globals.update({
            'List': typing.List, 'Dict': typing.Dict, 'Tuple': typing.Tuple,
            'Optional': typing.Optional, 'Union': typing.Union, 'Any': typing.Any,
        })

        # Execute solution
        try:
            exec(solution, exec_globals)
        except Exception:
            return 0.0  # Execution failed, return 0

        fn_name = test_cases.get("fn_name")
        if not fn_name or fn_name not in exec_globals:
            return 0.0  # Function not found, return 0

        # Run assert tests
        assert_cases = test_cases.get("assert_case", [])
        if isinstance(assert_cases, str):
            assert_cases = [assert_cases]

        results = []
        for case_group in assert_cases:
            if not isinstance(case_group, str):
                continue
            asserts = [line.strip() for line in case_group.split('\n') if line.strip()]
            for assert_stmt in asserts:
                if not assert_stmt.startswith("assert "):
                    continue

                try:
                    exec(assert_stmt, exec_globals)
                    results.append(True)
                except Exception:
                    results.append(False)

        # Return score based on results
        if not continuous:
            return 1.0 if all(results) else 0.0
        else:
            return sum(results) / len(results) if results else 0.0

    except Exception:
        return 0.0


def _validate_with_expression(solution: str, test_cases: dict, continuous: bool):
    """Validate code using logical expressions like 'fn() or fn() != fn()'"""
    try:
        # Create restricted globals to prevent network access
        exec_globals = {
            '__builtins__': {
                '__import__': __import__,
                'abs': abs, 'all': all, 'any': any, 'bin': bin, 'bool': bool, 'bytearray': bytearray,
                'bytes': bytes, 'callable': callable, 'chr': chr, 'classmethod': classmethod,
                'complex': complex, 'dict': dict, 'dir': dir, 'divmod': divmod, 'enumerate': enumerate,
                'filter': filter, 'float': float, 'format': format, 'frozenset': frozenset,
                'getattr': getattr, 'globals': globals, 'hasattr': hasattr, 'hash': hash,
                'help': help, 'hex': hex, 'id': id, 'int': int, 'isinstance': isinstance,
                'issubclass': issubclass, 'iter': iter, 'len': len, 'list': list, 'locals': locals,
                'map': map, 'max': max, 'memoryview': memoryview, 'min': min, 'next': next,
                'object': object, 'oct': oct, 'open': open, 'ord': ord, 'pow': pow, 'property': property,
                'range': range, 'repr': repr, 'reversed': reversed, 'round': round, 'set': set,
                'setattr': setattr, 'slice': slice, 'sorted': sorted, 'staticmethod': staticmethod,
                'str': str, 'sum': sum, 'super': super, 'tuple': tuple, 'type': type, 'vars': vars,
                'zip': zip, 'print': print, 'Exception': Exception, 'ValueError': ValueError,
                'TypeError': TypeError, 'IndexError': IndexError, 'KeyError': KeyError,
                'AttributeError': AttributeError, 'ImportError': ImportError, 'NameError': NameError,
                'math': __import__('math'), 'random': __import__('random'), 'collections': __import__('collections'),
                'itertools': __import__('itertools'), 'functools': __import__('functools'), 'operator': __import__('operator'),
                're': __import__('re'), 'json': __import__('json'), 'datetime': __import__('datetime'), 'time': __import__('time'),
                'typing': __import__('typing'),
            }
        }

        # Add common type annotations to globals
        import typing
        exec_globals.update({
            'List': typing.List, 'Dict': typing.Dict, 'Tuple': typing.Tuple,
            'Optional': typing.Optional, 'Union': typing.Union, 'Any': typing.Any,
        })

        # Execute solution
        try:
            exec(solution, exec_globals)
        except Exception:
            return 0.0  # Execution failed, return 0

        # Evaluate expressions
        outputs = test_cases.get("outputs", [])
        results = []

        for expr in outputs:
            try:
                result = eval(expr, exec_globals)
                results.append(bool(result))
            except Exception:
                results.append(False)

        # Return score based on results
        if not continuous:
            return 1.0 if all(results) else 0.0
        else:
            return sum(results) / len(results) if results else 0.0

    except Exception:
        return 0.0


def test_network_restriction():
    """Test function to verify network access restrictions work."""
    # Test code that tries to access network
    test_code = '''
import urllib.request
def test_func():
    return urllib.request.urlopen("http://example.com").read()
'''

    test_cases = {
        "call_type": "assert",
        "fn_name": "test_func",
        "assert_case": ["assert test_func() is not None"]
    }

    result = _validate_with_assert(test_code, test_cases, continuous=False)
    print(f"Network restriction test result: {result}")
    return result


if __name__ == "__main__":
    test_network_restriction()
