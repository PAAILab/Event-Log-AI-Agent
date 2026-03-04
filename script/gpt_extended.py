from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from openai import OpenAI

try:
    from openai import RateLimitError, APIStatusError
except Exception:
    RateLimitError = Exception
    APIStatusError = Exception


REQUIRED_COLUMNS = [
    "row_id",
    "error_flag",
    "error_types",  # Removed error_type_primary
    "error_confidence",
    "error_evidence",
]


# -----------------------------
# Simplified LLM logging
# -----------------------------
class LLMCallLogger:
    """
    Logs only iteration number, errors (if any), and metrics per iteration.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Initialize with empty array
        self.path.write_text("[]", encoding="utf-8")

    def log(self, record: Dict[str, Any]) -> None:
        """Append record to JSON array"""
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        data.append(record)
        
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def call_llm_with_logging(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
    logger: Optional[LLMCallLogger] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Calls chat.completions (logging removed from here, done at iteration level)"""
    meta = meta or {}

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content

    return text


def parse_ground_truth_label(label_str) -> set:
    """Parse ground truth label to extract error types"""
    if pd.isna(label_str) or str(label_str).strip() == "":
        return set()

    label_lower = str(label_str).lower()
    error_types = set()

    if "autoformbased" in label_lower or "form-based" in label_lower or "form_based" in label_lower or "formbased" in label_lower:
        error_types.add("form-based")
    if "polluted" in label_lower:
        error_types.add("polluted")
    if "distorted" in label_lower:
        error_types.add("distorted")
    if "synonymous" in label_lower:
        error_types.add("synonymous")
    if "collateral" in label_lower:
        error_types.add("collateral")
    if "homonymous" in label_lower:
        error_types.add("homonymous")
    if "empty" in label_lower:
        error_types.add("empty")

    return error_types


def compute_metrics_from_output(output_csv: Path, ground_truth_csv: Path) -> Dict[str, Any]:
    """
    Compute metrics by comparing LLM output with ground truth in THREE styles:
    
    1. STRICT: Exact match - all GT error types must exactly match all detected types
    2. MODERATE: Exact subset match - detected types must exactly match GT (no more, no less)
    3. GENEROUS: Any overlap - at least one error type matches
    
    Ground truth CSV should have a 'label' column with error types.
    LLM output CSV should have 'error_flag' and 'error_types_all' columns.
    """
    # Load ground truth
    df_gt = pd.read_csv(ground_truth_csv)
    if 'label' not in df_gt.columns:
        return {
            "error": "Ground truth CSV missing 'label' column",
            "metrics_strict": None,
            "metrics_moderate": None,
            "metrics_generous": None
        }
    
    # Load LLM output
    df_out = pd.read_csv(output_csv)
    if 'error_flag' not in df_out.columns or 'error_types' not in df_out.columns:
        return {
            "error": "Output CSV missing required columns",
            "metrics_strict": None,
            "metrics_moderate": None,
            "metrics_generous": Noneis
        }
    
    # Ensure both dataframes have same length and alignment
    if len(df_gt) != len(df_out):
        return {
            "error": f"Row count mismatch: GT={len(df_gt)}, Output={len(df_out)}",
            "metrics_strict": None,
            "metrics_moderate": None,
            "metrics_generous": None
        }
    
    def normalize_error_type(t: str) -> str:
        if not t:
            return ""
        t = t.lower().strip()
        t = t.replace("_", "-")
        t = t.replace("formbased", "form-based")
        return t

    # Parse ground truth labels
    df_gt['gt_error_types'] = df_gt['label'].apply(lambda x: {normalize_error_type(t) for t in parse_ground_truth_label(x)})
    df_gt['gt_has_error'] = df_gt['gt_error_types'].apply(lambda x: len(x) > 0)
    
    # Get predictions
    y_true = df_gt['gt_has_error']
    y_pred = df_out['error_flag'].fillna(False).astype(bool)
    
    CANONICAL_MAP = {
        "form_based": "form-based",
        "formbased": "form-based",
        "form-based": "form-based",
        "autoformbased": "form-based", 
        "distorted": "distorted",
        "polluted": "polluted", 
        "homonymous": "homonymous",
        "synonymous": "synonymous",
        "collateral": "collateral", 
        "empty": "empty"
    }
    
    def canonicalize(t: str) -> str:
        if not t or pd.isna(t):
            return ""
        t = t.lower().strip().replace("_", "-")
        return CANONICAL_MAP.get(t, t)
    
    # Extract predicted error types for each row
    def get_predicted_types(idx):
        pred_types = set()
        all_types = str(df_out.iloc[idx].get("error_types", ""))
        for t in all_types.split("|"):
            canonical = canonicalize(t)
            if canonical:
                pred_types.add(canonical)
        return pred_types
    
    # STYLE 1: STRICT - Exact match (all GT types must match all detected types)
    def is_correct_detection_strict(idx):
        if not y_pred.iloc[idx]:
            return False
        if not y_true.iloc[idx]:
            return False
        gt_types = {canonicalize(t) for t in df_gt.iloc[idx]["gt_error_types"]}
        gt_types.discard("")
        pred_types = get_predicted_types(idx)
        # Exact match: sets must be identical
        return gt_types == pred_types
    
    # STYLE 2: MODERATE - Predicted must exactly match GT (no more, no less)
    # If GT is "distorted polluted" and LLM detects "distorted" → FALSE
    # If GT is "distorted polluted" and LLM detects "distorted polluted form-based" → FALSE
    # If GT is "distorted polluted" and LLM detects "distorted polluted" → TRUE
    def is_correct_detection_moderate(idx):
        if not y_pred.iloc[idx]:
            return False
        if not y_true.iloc[idx]:
            return False
        gt_types = {canonicalize(t) for t in df_gt.iloc[idx]["gt_error_types"]}
        gt_types.discard("")
        pred_types = get_predicted_types(idx)
        # Must be exact match
        return pred_types.issubset(gt_types)
    
    # STYLE 3: GENEROUS - Any overlap (at least one type matches)
    def is_correct_detection_generous(idx):
        if not y_pred.iloc[idx]:
            return False
        if not y_true.iloc[idx]:
            return False
        gt_types = {canonicalize(t) for t in df_gt.iloc[idx]["gt_error_types"]}
        gt_types.discard("")
        pred_types = get_predicted_types(idx)
        # At least one match
        return len(gt_types & pred_types) > 0
    
    # Calculate correct detections for each style
    correct_strict = [is_correct_detection_strict(i) for i in range(len(df_gt))]
    correct_moderate = [is_correct_detection_moderate(i) for i in range(len(df_gt))]
    correct_generous = [is_correct_detection_generous(i) for i in range(len(df_gt))]
    
    # Helper function to calculate metrics for a given correctness list
    def calculate_metrics_for_style(correct_detections, style_name):
        correct = pd.Series(correct_detections, index=df_gt.index)
        
        tp = int((y_pred & y_true & correct).sum())
        fp = int((y_pred & (~correct | ~y_true)).sum())
        fn = int((y_true & (~y_pred | ~correct)).sum())
        tn = int((~y_pred & ~y_true).sum())
        
        total = len(df_gt)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Error type distribution
        error_df = df_out[df_out["error_flag"] == True]
        type_distribution = {}
        if not error_df.empty:
            for _, row in error_df.iterrows():
                types = get_predicted_types(row.name)
                for t in types:
                    type_distribution[t] = type_distribution.get(t, 0) + 1
        
        return {
            "style": style_name,
            "confusion_matrix": {
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn
            },
            "metrics": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "accuracy": round(accuracy, 4),
                "false_positive_rate": round(fpr, 4),
                "false_negative_rate": round(fnr, 4),
            },
            "counts": {
                "total_rows": total,
                "ground_truth_errors": int(y_true.sum()),
                "predicted_errors": int(y_pred.sum()),
                "correctly_detected": tp
            },
            "error_type_distribution": type_distribution
        }
    
    # Calculate metrics for all three styles
    metrics_strict = calculate_metrics_for_style(correct_strict, "STRICT")
    metrics_moderate = calculate_metrics_for_style(correct_moderate, "MODERATE")
    metrics_generous = calculate_metrics_for_style(correct_generous, "GENEROUS")
    
    return {
        "metrics_strict": metrics_strict,
        "metrics_moderate": metrics_moderate,
        "metrics_generous": metrics_generous,
    }


@dataclass
class DetectionContext:
    """Context for 5-step detection pipeline"""

    dataset_stem: str
    input_csv: Path
    ground_truth_csv: Path
    output_csv: Path
    summary_path: Path
    rules_path: Path

    detection_summary: Dict[str, Any] = field(default_factory=dict)
    analysis: Dict[str, Any] = field(default_factory=dict)
    improvement_history: List[Dict[str, Any]] = field(default_factory=list)

    fixed_csv: Optional[Path] = None
    evaluation_metrics: Dict[str, Any] = field(default_factory=dict)
    confusion_analysis: Dict[str, Any] = field(default_factory=dict)
    evaluation_path: Optional[Path] = None
    post_fix_metrics: Dict[str, Any] = field(default_factory=dict)
    post_fix_evaluation_path: Optional[Path] = None

    # Simplified logging
    llm_log_path: Optional[Path] = None
    llm_logger: Optional[LLMCallLogger] = None


def slugify(stem: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "data").is_dir():
            return p
    cwd = Path.cwd().resolve()
    if (cwd / "data").is_dir():
        return cwd
    raise RuntimeError("Could not locate repo root with 'data/' directory")


def load_prompt_text(repo_root: Path, prompt_rel_path: str) -> str:
    prompt_path = repo_root / prompt_rel_path
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def safe_parse_json(text: str) -> Dict[str, Any]:
    """Parse JSON robustly, extracting first {...} block if needed."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    json_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_block_match:
        try:
            return json.loads(json_block_match.group(1))
        except json.JSONDecodeError:
            pass

    brace_count = 0
    start_idx = -1

    for i, char in enumerate(text):
        if char == "{":
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                try:
                    return json.loads(text[start_idx : i + 1])
                except json.JSONDecodeError:
                    start_idx = -1

    raise ValueError("No valid JSON found in response")


def extract_python_code(text: str) -> str:
    """Extract Python code from various formats (markdown, raw, etc.)."""
    python_block_match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if python_block_match:
        return python_block_match.group(1).strip()

    code_block_match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if code_block_match:
        code = code_block_match.group(1).strip()
        if any(keyword in code for keyword in ["import ", "def ", "class ", "pd.", "df = "]):
            return code

    if any(keyword in text for keyword in ["import pandas", "import numpy", "df = pd.read_csv"]):
        return text.strip()

    return ""


class DetectorAgentWithLoop:
    SYSTEM_PROMPT = """You are an aggressive error detection expert. Your goal is to find real errors in data.

CRITICAL RULES:
1. Be thorough - check every column for potential issues
2. Be aggressive - real data has 5-20% error rates
3. Calculate confidence based on evidence strength (not fixed values)
4. Use diverse error types (duplicates, nulls, format, range, logic)
5. Write specific descriptions (what, why, values, IDs)

If previous attempt failed, read the feedback and implement the specific fixes requested."""

    def __init__(self, client: OpenAI, model: str, prompt_template: str, scripts_dir: Path):
        self.client = client
        self.model = model
        self.prompt_template = prompt_template
        self.scripts_dir = scripts_dir
        self.min_confidence = 0.65
        self.min_error_types = 3
        self.max_iterations = 15
        self.script_timeout_seconds = 240  # 4 minutes

    def execute(self, context: DetectionContext) -> DetectionContext:
        print(f"\n[STEP 1/5] Detection - {context.dataset_stem}")

        iteration = 0
        improvement_history: List[Dict[str, Any]] = []
        best_result: Optional[Dict[str, Any]] = None
        best_score = -1.0

        while iteration < self.max_iterations:
            iteration += 1
            script_path: Optional[Path] = None
            iteration_record = {
                "iteration": iteration,
                "error": None,
                "metrics_strict": None,
                "metrics_moderate": None,
                "metrics_generous": None,
                "execution_time_seconds": None
            }

            try:
                quality_feedback = self._get_aggressive_feedback(improvement_history, iteration)
                prompt = self._build_prompt(context, quality_feedback, iteration)

                response = self._call_llm(prompt, context, iteration)

                python_code = ""
                try:
                    payload = safe_parse_json(response)
                    python_code = payload.get("python_code", "")
                except ValueError:
                    pass

                if not python_code.strip():
                    python_code = extract_python_code(response)

                if not python_code.strip():
                    iteration_record["error"] = "Could not extract Python code"
                    improvement_history.append(iteration_record)
                    if context.llm_logger:
                        context.llm_logger.log(iteration_record)
                    continue

                script_path = self._write_script(context.dataset_stem, {"python_code": python_code}, iteration)

                # Run script with timeout and time tracking
                start_time = time.time()
                result = self._run_script_with_timeout(script_path, self.script_timeout_seconds)
                execution_time = time.time() - start_time
                iteration_record["execution_time_seconds"] = round(execution_time, 2)

                if result.returncode != 0:
                    raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)

                self._validate_output(context.output_csv)

                # Compute metrics against ground truth (returns all 3 styles)
                metrics_result = compute_metrics_from_output(output_csv=context.output_csv, ground_truth_csv=context.ground_truth_csv)

                
                if "error" in metrics_result:
                    iteration_record["error"] = metrics_result["error"]
                else:
                    # Store all three metric styles
                    iteration_record["metrics_strict"] = metrics_result.get("metrics_strict")
                    iteration_record["metrics_moderate"] = metrics_result.get("metrics_moderate")
                    iteration_record["metrics_generous"] = metrics_result.get("metrics_generous")
                
                improvement_history.append(iteration_record)
                
                # Log to JSON file
                if context.llm_logger:
                    context.llm_logger.log(iteration_record)

                # Use GENEROUS metrics for iteration decisions (for backward compatibility)
                if iteration_record.get("metrics_generous"):
                    m = iteration_record["metrics_generous"]["metrics"]
                    print(
                        f"  Iteration {iteration}: "
                        f"P={m['precision']:.3f}, R={m['recall']:.3f}, F1={m['f1_score']:.3f}, "
                        f"Errors={iteration_record['metrics_generous']['counts']['predicted_errors']}, "
                        f"Time={execution_time:.1f}s"
                    )

                    # Calculate quality score using generous metrics
                    quality_metrics = self._convert_to_quality_metrics(iteration_record["metrics_generous"])
                    score = self._calculate_quality_score(quality_metrics)
                    
                    if score > best_score:
                        best_score = score
                        best_result = {"iteration": iteration, "metrics_result": metrics_result, "script_path": script_path}

                    if self._is_quality_excellent(quality_metrics, iteration):
                        print(f"  Quality achieved at iteration {iteration}")
                        context.improvement_history = improvement_history
                        return context

            except subprocess.TimeoutExpired as e:
                execution_time = self.script_timeout_seconds
                iteration_record["execution_time_seconds"] = execution_time
                error_msg = f"Script execution exceeded timeout of {self.script_timeout_seconds}s ({self.script_timeout_seconds/60:.1f} minutes). The script is too slow and needs optimization."
                iteration_record["error"] = error_msg
                improvement_history.append(iteration_record)
                
                if context.llm_logger:
                    context.llm_logger.log(iteration_record)
                
                print(f"  Iteration {iteration}: TIMEOUT after {execution_time:.1f}s")

            except subprocess.CalledProcessError as e:
                error_msg = self._parse_execution_error(e, script_path if script_path else None)
                iteration_record["error"] = error_msg[:500]  # Truncate for JSON
                improvement_history.append(iteration_record)
                
                if context.llm_logger:
                    context.llm_logger.log(iteration_record)

            except Exception as e:
                iteration_record["error"] = str(e)
                improvement_history.append(iteration_record)
                
                if context.llm_logger:
                    context.llm_logger.log(iteration_record)

        if best_result:
            print(f"  Using best result from iteration {best_result['iteration']} (score: {best_score:.3f})")

        context.improvement_history = improvement_history
        return context

    def _convert_to_quality_metrics(self, metrics_result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert metrics result to quality metrics format for quality checks"""
        counts = metrics_result.get("counts", {})
        metrics = metrics_result.get("metrics", {})
        type_dist = metrics_result.get("error_type_distribution", {})
        
        total_rows = counts.get("total_rows", 1)
        predicted_errors = counts.get("predicted_errors", 0)
        error_rate = predicted_errors / total_rows if total_rows > 0 else 0.0
        
        # Build issues list
        issues = []
        if error_rate >= 0.99:
            issues.append(f"CRITICAL: {error_rate:.1%} flagged - detection logic is broken")
        elif predicted_errors == 0:
            issues.append("CRITICAL: Zero errors detected")
        elif error_rate > 0.50:
            issues.append(f"CRITICAL: {error_rate:.1%} flagged - too aggressive (>90%)")
        elif error_rate < 0.02:
            issues.append(f"CRITICAL: Only {error_rate:.1%} flagged - too conservative (<15%)")
        elif 0.30 < error_rate <= 0.50:
            issues.append(f"WARNING: {error_rate:.1%} flagged - on the high side (70-90%)")
        elif 0.02 <= error_rate < 0.05:
            issues.append(f"WARNING: {error_rate:.1%} flagged - on the low side (15-30%)")
        
        return {
            "total_rows": total_rows,
            "errors_detected": predicted_errors,
            "error_rate": error_rate,
            "avg_confidence": 0.75,  # Default since we don't track this anymore
            "unique_error_types": len(type_dist),
            "error_type_distribution": type_dist,
            "issues": issues,
            "critical_issues": [i for i in issues if "CRITICAL" in i],
        }

    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        score = 0.0
        error_rate = float(metrics.get("error_rate", 0.0))
        if 0.15 <= error_rate <= 0.90:
            if 0.30 <= error_rate <= 0.70:
                score = 1.0
            elif 0.15 <= error_rate < 0.30:
                score = 0.5 + 0.5 *((error_rate - 0.15) / 0.15)
            else:
                score = 1.0 - 0.5 * ((error_rate - 0.70) / 0.20)
        elif error_rate < 0.15:
            score = max(0.0, error_rate / 0.15 * 0.5)
        else: 
            score = max(0.0, 0.3 * (1.0 - (error_rate - 0.90) / 0.10))
        return float(score)

    def _is_quality_excellent(self, metrics: Dict[str, Any], iteration: int) -> bool:
        error_rate = metrics.get("error_rate", 0)
        if error_rate >= 0.99:
            return False 
        if error_rate > 0.50 or error_rate < 0.02:
            return False 
        if iteration <= 3:
            return (int(metrics.get("errors_detected", 0)) > 0 and len(metrics.get("critical_issues", [])) == 0)

        if iteration <= 8:
            return len(metrics.get("issues", [])) <= 1

        return len(metrics.get("issues", [])) <=1

    def _get_aggressive_feedback(self, history: List[Dict[str, Any]], iteration: int) -> str:
        if not history:
            return ""

        latest = history[-1]
        exec_time = latest.get("execution_time_seconds")

        if "error" in latest and latest["error"]:
            error_msg = latest["error"]
            
            # Check if it's a timeout error
            if "exceeded timeout" in error_msg.lower():
                feedback = (
                    f"\n\n{'='*70}\nPREVIOUS ATTEMPT TIMED OUT\n{'='*70}\n"
                    f"Script execution took longer than {self.script_timeout_seconds}s (4 minutes).\n\n"
                    f"PERFORMANCE OPTIMIZATION REQUIRED:\n"
                    f"Your code is too slow. Common causes:\n"
                    f"1. Using df.iterrows() - Replace with vectorized operations\n"
                    f"2. Nested loops with O(n²) complexity - Optimize algorithm\n"
                    f"3. Too many canonical comparisons - Limit to top N most frequent\n"
                    f"4. Slow string similarity (difflib) - Use faster alternatives or limit comparisons\n\n"
                    f"REQUIRED FIXES:\n"
                    f"- Remove or optimize any df.iterrows() loops\n"
                    f"- Limit canonical activity list to top 50-100 most frequent\n"
                    f"- Use vectorized pandas operations instead of Python loops\n"
                    f"- Avoid nested loops over large datasets\n\n"
                    f"Target: Complete execution in under 60 seconds.\n"
                )
                return feedback
            else:
                return (
                    f"\n\n{'='*70}\nPREVIOUS ATTEMPT FAILED\n{'='*70}\n"
                    f"{error_msg}\n\nFIX THIS ERROR IMMEDIATELY."
                )

        # Use generous metrics for feedback
        metrics_result = latest.get("metrics_generous")
        if not metrics_result:
            return ""

        # Convert to quality metrics
        quality_metrics = self._convert_to_quality_metrics(metrics_result)
        issues = quality_metrics.get("issues", [])
    
        if not issues:
            return ""

        feedback = f"\n\n{'='*70}\nITERATION {iteration}: QUALITY INSUFFICIENT\n{'='*70}\n"
        feedback += "\nCURRENT STATE:\n"
        feedback += f"  - Errors: {quality_metrics.get('errors_detected', 0):,} ({quality_metrics.get('error_rate', 0.0):.2%})\n"
        feedback += f"  - Error Types: {quality_metrics.get('unique_error_types', 0)}\n"
        feedback += f"  - Type Distribution: {quality_metrics.get('error_type_distribution', {})}\n"
        
        # Add execution time info
        if exec_time is not None:
            feedback += f"  - Execution Time: {exec_time:.1f}s"
            if exec_time > 120:
                feedback += " ⚠️ SLOW (target: <60s)"
            feedback += "\n"
        
        # Add metrics from actual evaluation
        m = metrics_result.get("metrics", {})
        feedback += f"  - Precision: {m.get('precision', 0):.3f}\n"
        feedback += f"  - Recall: {m.get('recall', 0):.3f}\n"
        feedback += f"  - F1 Score: {m.get('f1_score', 0):.3f}\n"

        feedback += "\n" + "="*70 + "\n"
        feedback += "ISSUES TO FIX:\n"
        feedback += "="*70 + "\n"
    
        for i, issue in enumerate(issues, 1):
            feedback += f"\n{i}. {issue}\n"
        
        # Add performance warning if slow
        if exec_time is not None and exec_time > 120:
            feedback += "\n" + "="*70 + "\n"
            feedback += "PERFORMANCE WARNING:\n"
            feedback += "="*70 + "\n"
            feedback += f"Script took {exec_time:.1f}s (>{exec_time/60:.1f} min). Optimize for faster execution:\n"
            feedback += "- Limit canonical activity list to top 50-100 most frequent\n"
            feedback += "- Use vectorized pandas operations instead of loops\n"
            feedback += "- Avoid df.iterrows() - use apply() or vectorized methods\n"

        return feedback

    def _parse_execution_error(self, e: subprocess.CalledProcessError, script_path: Optional[Path] = None) -> str:
        stderr = e.stderr or ""
        stdout = e.stdout or ""

        feedback = "=" * 70 + "\n"
        feedback += "SCRIPT EXECUTION FAILED\n"
        feedback += "=" * 70 + "\n\n"

        if "Traceback" in stderr:
            error_lines = stderr.strip().split("\n")[-12:]
            feedback += "ERROR TRACEBACK:\n"
            feedback += "\n".join(error_lines) + "\n\n"

        return feedback

    def _build_prompt(self, context: DetectionContext, quality_feedback: str, iteration: int) -> str:
        base_prompt = (
            self.prompt_template.replace("{INPUT_CSV_PATH}", str(context.input_csv.as_posix()))
            .replace("{OUTPUT_CSV_PATH}", str(context.output_csv.as_posix()))
            .replace("{DATASET_STEM}", context.dataset_stem)
        )
        if iteration > 1 and quality_feedback:
            base_prompt += quality_feedback
        return base_prompt

    def _call_llm(self, prompt: str, context: DetectionContext, iteration: int) -> str:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return call_llm_with_logging(
            client=self.client,
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=None,
            logger=context.llm_logger,
            meta={"step": "detector", "dataset": context.dataset_stem, "iteration": iteration},
        )

    def _write_script(self, dataset_stem: str, payload: Dict[str, Any], iteration: int) -> Path:
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        code = payload.get("python_code", "")
        if not code.strip():
            raise RuntimeError("Model returned empty python_code")
        out_path = self.scripts_dir / f"detect_{dataset_stem}_v{iteration}.py"
        out_path.write_text(code, encoding="utf-8")
        return out_path

    def _run_script(self, script_path: Path) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)

    def _run_script_with_timeout(self, script_path: Path, timeout_seconds: int) -> subprocess.CompletedProcess:
        """Run script with timeout. Raises subprocess.TimeoutExpired if timeout is exceeded."""
        return subprocess.run(
            [sys.executable, str(script_path)], 
            capture_output=True, 
            text=True,
            timeout=timeout_seconds
        )

    def _validate_output(self, out_csv: Path) -> None:
        df = pd.read_csv(out_csv)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise RuntimeError(f"Output missing columns: {missing}")


class SummaryAgent:
    SYSTEM_PROMPT = "You are a data quality expert analyzing error detection results. Return your analysis as valid JSON only."

    def __init__(self, client: OpenAI, model: str, prompt_template: str):
        self.client = client
        self.model = model
        self.prompt_template = prompt_template

    def execute(self, context: DetectionContext) -> DetectionContext:
        print("\n[STEP 2/5] Summary Analysis")

        df = pd.read_csv(context.output_csv)
        summary = self._generate_summary(df)
        context.detection_summary = summary

        prompt = self._build_prompt(context, summary)
        response = self._call_llm(prompt, context)
        analysis = safe_parse_json(response)
        context.analysis = analysis

        self._save_report(context, summary, analysis)
        return context

    def _generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        error_df = df[df["error_flag"] == True]
        
        # Build type counts from error_types_all instead of error_type_primary
        type_counts = {}
        if not error_df.empty:
            for _, row in error_df.iterrows():
                types = str(row.get("error_types_all", "")).split("|")
                for t in types:
                    t = t.strip()
                    if t:
                        type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_rows": int(len(df)),
            "errors_detected": int(len(error_df)),
            "error_percentage": round(len(error_df) / len(df) * 100, 2) if len(df) > 0 else 0.0,
            "error_type_counts": type_counts,
        }

    def _build_prompt(self, context: DetectionContext, summary: Dict[str, Any]) -> str:
        summary_text = json.dumps(summary, indent=2)
        return (
            self.prompt_template.replace("{DATASET_STEM}", context.dataset_stem)
            .replace("{INPUT_CSV_PATH}", str(context.input_csv.as_posix()))
            .replace("{DETECTION_OUTPUT_CSV_PATH}", str(context.output_csv.as_posix()))
            .replace("{DETECTION_SUMMARY}", summary_text)
        )

    def _call_llm(self, prompt: str, context: DetectionContext) -> str:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return call_llm_with_logging(
            client=self.client,
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=None,
            logger=context.llm_logger,
            meta={"step": "summary", "dataset": context.dataset_stem},
        )

    def _save_report(self, context: DetectionContext, summary: Dict[str, Any], analysis: Dict[str, Any]) -> None:
        report = {
            "dataset": context.dataset_stem,
            "input_csv": str(context.input_csv),
            "output_csv": str(context.output_csv),
            "summary": summary,
            "analysis": analysis,
        }
        context.summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


class RuleGeneratorAgent:
    SYSTEM_PROMPT = "You are an expert at defining formal validation rules for data quality. Return your rules as valid JSON only."

    def __init__(self, client: OpenAI, model: str, prompt_template: str):
        self.client = client
        self.model = model
        self.prompt_template = prompt_template

    def execute(self, context: DetectionContext) -> DetectionContext:
        print("\n[STEP 3/5] Rule Generation")

        prompt = self._build_prompt(context)
        response = self._call_llm(prompt, context)
        rules = safe_parse_json(response)
        self._save_rules(context, rules)

        return context

    def _build_prompt(self, context: DetectionContext) -> str:
        summary_text = json.dumps(context.detection_summary, indent=2)
        analysis_text = json.dumps(context.analysis, indent=2)

        return (
            self.prompt_template.replace("{DATASET_STEM}", context.dataset_stem)
            .replace("{DETECTION_SUMMARY}", summary_text)
            .replace("{ANALYSIS}", analysis_text)
        )

    def _call_llm(self, prompt: str, context: DetectionContext) -> str:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return call_llm_with_logging(
            client=self.client,
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=None,
            logger=context.llm_logger,
            meta={"step": "rule_generator", "dataset": context.dataset_stem},
        )

    def _save_rules(self, context: DetectionContext, rules: Dict[str, Any]) -> None:
        context.rules_path.write_text(json.dumps(rules, indent=2), encoding="utf-8")


class FixerAgent:
    SYSTEM_PROMPT = "You are an expert at generating Python data fixing scripts. Generate complete scripts that fix errors based on provided rules."

    def __init__(self, client: OpenAI, model: str, prompt_template: str, scripts_dir: Path):
        self.client = client
        self.model = model
        self.prompt_template = prompt_template
        self.scripts_dir = scripts_dir

    def execute(self, context: DetectionContext) -> DetectionContext:
        print("\n[STEP 4/5] Fixing Errors")

        if not context.rules_path.exists():
            return context

        rules = json.loads(context.rules_path.read_text(encoding="utf-8"))
        prompt = self._build_prompt(context, rules)
        response = self._call_llm(prompt, context)

        try:
            payload = safe_parse_json(response)
            python_code = payload.get("python_code", "")
        except ValueError:
            python_code = extract_python_code(response)
            if not python_code:
                return context

        script_path = self._write_script(context.dataset_stem, python_code)

        try:
            result = self._run_script(script_path)

            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)

            fixed_csv = context.output_csv.parent / f"{context.input_csv.stem}.fixed.csv"
            if fixed_csv.exists():
                context.fixed_csv = fixed_csv

        except subprocess.CalledProcessError:
            pass

        return context

    def _build_prompt(self, context: DetectionContext, rules: Dict[str, Any]) -> str:
        rules_text = json.dumps(rules, indent=2)
        summary_text = json.dumps(context.detection_summary, indent=2)

        detected_csv = str(context.output_csv.as_posix())
        fixed_csv = str((context.output_csv.parent / f"{context.input_csv.stem}.fixed.csv").as_posix())

        return (
            self.prompt_template.replace("{DETECTED_CSV_PATH}", detected_csv)
            .replace("{FIXED_CSV_PATH}", fixed_csv)
            .replace("{RULES}", rules_text)
            .replace("{DETECTION_SUMMARY}", summary_text)
            .replace("{DATASET_STEM}", context.dataset_stem)
        )

    def _call_llm(self, prompt: str, context: DetectionContext) -> str:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return call_llm_with_logging(
            client=self.client,
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=None,
            logger=context.llm_logger,
            meta={"step": "fixer", "dataset": context.dataset_stem},
        )

    def _write_script(self, dataset_stem: str, code: str) -> Path:
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.scripts_dir / f"fix_{dataset_stem}.py"
        out_path.write_text(code, encoding="utf-8")
        return out_path

    def _run_script(self, script_path: Path) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)


class EvaluatorAgent:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def execute(self, context: DetectionContext) -> DetectionContext:
        print("\n[STEP 5/5] Evaluation")

        # Compute final metrics using ground truth (returns all 3 styles)
        metrics_result = compute_metrics_from_output(output_csv=context.output_csv, ground_truth_csv=context.ground_truth_csv)

        
        if "error" not in metrics_result:
            context.evaluation_metrics = metrics_result
            self._save_evaluation_report(context, metrics_result)

        if context.fixed_csv and context.fixed_csv.exists():
            df_fixed = pd.read_csv(context.fixed_csv)
            if "label" in df_fixed.columns:
                post_fix_metrics = self._calculate_post_fix_metrics(df_fixed, context.ground_truth_csv)
                context.post_fix_metrics = post_fix_metrics
                self._save_post_fix_evaluation(context, post_fix_metrics)

        return context

    def _calculate_post_fix_metrics(self, df_fixed: pd.DataFrame, ground_truth_csv: Path) -> Dict[str, Any]:
        df_gt = pd.read_csv(ground_truth_csv)
        
        df_gt['gt_error_types'] = df_gt['label'].apply(parse_ground_truth_label)
        y_true_errors = df_gt['gt_error_types'].apply(lambda x: len(x) > 0)

        has_fix_applied = "fix_applied" in df_fixed.columns
        if has_fix_applied:
            fixed_mask = df_fixed["fix_applied"].notna() & (df_fixed["fix_applied"] != "")
        else:
            fixed_mask = pd.Series([False] * len(df_fixed), index=df_fixed.index)

        errors_needing_fix = int(y_true_errors.sum())
        errors_that_were_fixed = int((y_true_errors & fixed_mask).sum())

        precision_fix = errors_that_were_fixed / int(fixed_mask.sum()) if int(fixed_mask.sum()) > 0 else 0.0
        recall_fix = errors_that_were_fixed / errors_needing_fix if errors_needing_fix > 0 else 0.0
        f1_fix = (
            2 * (precision_fix * recall_fix) / (precision_fix + recall_fix)
            if (precision_fix + recall_fix) > 0
            else 0.0
        )

        clean_rows_incorrectly_fixed = int((~y_true_errors & fixed_mask).sum())
        error_rows_not_fixed = int((y_true_errors & ~fixed_mask).sum())

        return {
            "confusion_matrix": {
                "errors_fixed_correctly": errors_that_were_fixed,
                "clean_rows_modified": clean_rows_incorrectly_fixed,
                "errors_not_fixed": error_rows_not_fixed,
                "clean_rows_untouched": int((~y_true_errors & ~fixed_mask).sum()),
            },
            "metrics": {
                "precision": round(precision_fix, 4),
                "recall": round(recall_fix, 4),
                "f1_score": round(f1_fix, 4),
                "fix_rate": round(recall_fix, 4),
            },
            "counts": {
                "total_rows": int(len(df_fixed)),
                "ground_truth_errors": errors_needing_fix,
                "fixes_applied": int(fixed_mask.sum()),
                "errors_successfully_fixed": errors_that_were_fixed,
            },
        }

    def _save_evaluation_report(
        self,
        context: DetectionContext,
        metrics_result: Dict[str, Any],
    ) -> None:
        report = {
            "dataset": context.dataset_stem,
            "evaluation_type": "detection",
            "metrics": metrics_result,
            "improvement_history": context.improvement_history,
        }
        eval_path = context.output_csv.parent / f"{context.input_csv.stem}.evaluation.json"
        eval_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        context.evaluation_path = eval_path

    def _save_post_fix_evaluation(self, context: DetectionContext, post_fix_metrics: Dict[str, Any]) -> None:
        report = {
            "dataset": context.dataset_stem,
            "evaluation_type": "post_fix",
            "metrics": post_fix_metrics,
            "description": "Evaluation of how well fixes restored erroneous data to clean state",
        }
        post_fix_path = context.output_csv.parent / f"{context.input_csv.stem}.post_fix_evaluation.json"
        post_fix_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        context.post_fix_evaluation_path = post_fix_path


class DetectionPipeline:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        repo_root: Path,
        detector_prompt: str,
        summary_prompt: str,
        rules_prompt: str,
        fixer_prompt: str,
    ):
        scripts_dir = repo_root / "scripts"
        self.detector = DetectorAgentWithLoop(client, model, detector_prompt, scripts_dir)
        self.summarizer = SummaryAgent(client, model, summary_prompt)
        self.rule_generator = RuleGeneratorAgent(client, model, rules_prompt)
        self.fixer = FixerAgent(client, model, fixer_prompt, scripts_dir)
        self.evaluator = EvaluatorAgent(client, model)


    @staticmethod
    def resolve_ground_truth(no_label_csv: Path) -> Path:
        if "_noLabel" not in no_label_csv.stem:
            raise ValueError(f"Expected *_noLabel.csv, got {no_label_csv.name}")

        gt_name = no_label_csv.name.replace("_noLabel.csv", "_withLabel.csv")
        gt_path = no_label_csv.parent / gt_name

        if not gt_path.exists():
            raise FileNotFoundError(f"Missing ground truth file: {gt_path}")

        return gt_path



    def process_dataset(self, input_csv: Path, output_dir: Path) -> DetectionContext:
        dataset_stem = slugify(input_csv.stem)
        ground_truth_csv = self.resolve_ground_truth(input_csv)
        output_csv = output_dir / f"{input_csv.stem}.detected.csv"
        summary_path = output_dir / f"{input_csv.stem}.summary.json"
        rules_path = output_dir / f"{input_csv.stem}.rules.json"
        llm_log_path = output_dir / f"{input_csv.stem}.iterations.json"

        context = DetectionContext(
            dataset_stem=dataset_stem,
            input_csv=input_csv,
            ground_truth_csv=ground_truth_csv,
            output_csv=output_csv,
            summary_path=summary_path,
            rules_path=rules_path,
            llm_log_path=llm_log_path,
        )
        context.llm_logger = LLMCallLogger(llm_log_path)

        context = self.detector.execute(context)
        context = self.summarizer.execute(context)
        context = self.rule_generator.execute(context)
        context = self.fixer.execute(context)
        context = self.evaluator.execute(context)

        return context


def main() -> None:
    repo_root = find_repo_root(Path(__file__).resolve())
    data_dir = repo_root / "data"
    out_dir = repo_root / "data_detected"

    detector_prompt_path = "prompts/vanilla_examples_instructions.txt"
    summary_prompt_path = "prompts/summary_agent_prompt.txt"
    rules_prompt_path = "prompts/rule_generator_prompt.txt"
    fixer_prompt_path = "prompts/fixer_agent_prompt.txt"

    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    model = os.environ.get("OPENAI_MODEL", "gpt-5.2")
    client = OpenAI()

    detector_prompt = load_prompt_text(repo_root, detector_prompt_path)
    summary_prompt = load_prompt_text(repo_root, summary_prompt_path)
    rules_prompt = load_prompt_text(repo_root, rules_prompt_path)
    fixer_prompt = load_prompt_text(repo_root, fixer_prompt_path)

    pipeline = DetectionPipeline(
        client=client,
        model=model,
        repo_root=repo_root,
        detector_prompt=detector_prompt,
        summary_prompt=summary_prompt,
        rules_prompt=rules_prompt,
        fixer_prompt=fixer_prompt,
    )

    csv_files = sorted(data_dir.rglob("*_noLabel.csv"))
    if not csv_files:
        print(f"No CSV files in {data_dir}")
        return

    print("\n" + "=" * 70)
    print("Error Detection & Fixing Pipeline")
    print(f"Model: {model} | Datasets: {len(csv_files)}")
    print("=" * 70)

    all_results: List[Dict[str, Any]] = []

    for in_csv in csv_files:
        try:
            print("\n" + "=" * 70)
            print(f"DATASET: {in_csv.name}")
            print("=" * 70)

            context = pipeline.process_dataset(in_csv, out_dir)

            result = {
                "dataset": in_csv.name,
                "errors_detected": context.detection_summary.get("errors_detected", 0),
                "total_rows": context.detection_summary.get("total_rows", 0),
                "detection_metrics_strict": context.evaluation_metrics.get("metrics_strict", {}).get("metrics", {}),
                "detection_metrics_moderate": context.evaluation_metrics.get("metrics_moderate", {}).get("metrics", {}),
                "detection_metrics_generous": context.evaluation_metrics.get("metrics_generous", {}).get("metrics", {}),
                "post_fix_metrics": context.post_fix_metrics.get("metrics", {}) if context.post_fix_metrics else {},
            }
            all_results.append(result)

            print("\nOUTPUTS:")
            if context.fixed_csv:
                print(f"  - Cleaned Dataset: {context.fixed_csv}")
            print(f"  - Detected: {context.output_csv}")
            print(f"  - Summary: {context.summary_path}")
            print(f"  - Rules: {context.rules_path}")
            if context.evaluation_path:
                print(f"  - Evaluation: {context.evaluation_path}")
            if context.llm_log_path:
                print(f"  - Iteration Log: {context.llm_log_path}")

            if context.evaluation_metrics:
                print("\nDETECTION METRICS (ALL STYLES):")
                for style in ["strict", "moderate", "generous"]:
                    style_metrics = context.evaluation_metrics.get(f"metrics_{style}", {})
                    m = style_metrics.get("metrics", {})
                    cm = style_metrics.get("confusion_matrix", {})
                    c = style_metrics.get("counts", {})
                    
                    print(f"\n  {style.upper()}:")
                    print(f"    Precision: {m.get('precision', 0):.4f}")
                    print(f"    Recall:    {m.get('recall', 0):.4f}")
                    print(f"    F1 Score:  {m.get('f1_score', 0):.4f}")
                    print(f"    Accuracy:  {m.get('accuracy', 0):.4f}")
                    total_rows = c.get("total_rows", 0) or 1
                    pred_errs = c.get("predicted_errors", 0)
                    print(f"    Errors: {pred_errs} / {total_rows} ({pred_errs / total_rows * 100:.1f}%)")
                    print(
                        f"    TP: {cm.get('TP', 0)} | FP: {cm.get('FP', 0)} | "
                        f"FN: {cm.get('FN', 0)} | TN: {cm.get('TN', 0)}"
                    )

            if context.post_fix_metrics:
                m = context.post_fix_metrics.get("metrics", {})
                cm = context.post_fix_metrics.get("confusion_matrix", {})
                c = context.post_fix_metrics.get("counts", {})

                print("\nPOST-FIX METRICS:")
                print(f"  Precision: {m.get('precision', 0):.4f}")
                print(f"  Recall:    {m.get('recall', 0):.4f}")
                print(f"  F1 Score:  {m.get('f1_score', 0):.4f}")
                print(f"  Fix Rate:  {m.get('fix_rate', 0):.1%}")
                print(f"  Fixes: {c.get('fixes_applied', 0)} / {c.get('ground_truth_errors', 0)} errors")
                print(
                    f"  Correctly Fixed: {cm.get('errors_fixed_correctly', 0)} | "
                    f"Not Fixed: {cm.get('errors_not_fixed', 0)}"
                )

        except Exception as e:
            print(f"\nERROR: {e}")
            continue

    if all_results:
        print("\n" + "=" * 70)
        print(f"SUMMARY - {len(all_results)} DATASETS")
        print("=" * 70)

        total_datasets = len(all_results)
        
        # Calculate averages for each style
        for style in ["strict", "moderate", "generous"]:
            print(f"\n{style.upper()} METRICS:")
            
            avg_det_precision = sum(r[f"detection_metrics_{style}"].get("precision", 0) for r in all_results) / total_datasets
            avg_det_recall = sum(r[f"detection_metrics_{style}"].get("recall", 0) for r in all_results) / total_datasets
            avg_det_f1 = sum(r[f"detection_metrics_{style}"].get("f1_score", 0) for r in all_results) / total_datasets
            
            print(f"  Average Detection:")
            print(f"    Precision: {avg_det_precision:.4f}")
            print(f"    Recall:    {avg_det_recall:.4f}")
            print(f"    F1 Score:  {avg_det_f1:.4f}")

        results_with_fixes = [r for r in all_results if r["post_fix_metrics"]]
        if results_with_fixes:
            avg_fix_precision = sum(r["post_fix_metrics"].get("precision", 0) for r in results_with_fixes) / len(results_with_fixes)
            avg_fix_recall = sum(r["post_fix_metrics"].get("recall", 0) for r in results_with_fixes) / len(results_with_fixes)
            avg_fix_f1 = sum(r["post_fix_metrics"].get("f1_score", 0) for r in results_with_fixes) / len(results_with_fixes)
            
            print(f"\nAVERAGE POST-FIX METRICS ({len(results_with_fixes)} datasets):")
            print(f"  Precision: {avg_fix_precision:.4f}")
            print(f"  Recall:    {avg_fix_recall:.4f}")
            print(f"  F1 Score:  {avg_fix_f1:.4f}")

        print("\nINDIVIDUAL RESULTS (GENEROUS):")
        for r in all_results:
            dm = r["detection_metrics_generous"]
            fm = r["post_fix_metrics"]
            fix_str = f" | Fix F1: {fm.get('f1_score', 0):.4f}" if fm else ""
            print(f"  {r['dataset']:30s} | Det F1: {dm.get('f1_score', 0):.4f}{fix_str}")

        print("=" * 70)

    print(f"\nComplete - Outputs: {out_dir}")


if __name__ == "__main__":
    main()