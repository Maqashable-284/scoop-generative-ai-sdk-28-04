"""
Scoop AI Evals - Main Runner
Executes tests and generates reports
"""
import os
import sys
import json
import yaml
import uuid
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.judge import LLMJudge, EvalScore, create_judge
from evals.client import ScoopClient, ChatResponse, create_client
from evals.dashboard import save_html_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a single test"""
    test_id: str
    test_name: str
    input: str
    expected: str
    actual_response: str
    score: float
    passed: bool
    reason: str
    duration_seconds: float
    criteria_met: Dict[str, bool]


@dataclass
class SetResult:
    """Result of a test set"""
    set_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_rate: float
    avg_score: float
    results: List[TestResult]


@dataclass
class EvalReport:
    """Complete evaluation report"""
    timestamp: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    overall_pass_rate: float
    overall_avg_score: float
    set_results: List[SetResult]
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "overall_pass_rate": self.overall_pass_rate,
            "overall_avg_score": self.overall_avg_score,
            "set_results": [
                {
                    **asdict(sr),
                    "results": [asdict(r) for r in sr.results]
                }
                for sr in self.set_results
            ]
        }


class EvalRunner:
    """Main evaluation runner"""
    
    def __init__(self):
        self.client = create_client()
        self.judge = create_judge()
        self.test_cases = self._load_test_cases()
        
    def _load_test_cases(self) -> Dict:
        """Load test cases from YAML"""
        yaml_path = Path(__file__).parent / "test_cases.yaml"
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def run_single_test(
        self,
        test: Dict,
        session_id: Optional[str] = None
    ) -> TestResult:
        """Run a single test case"""
        import time
        start_time = time.time()
        
        test_id = test.get("id", "unknown")
        test_name = test.get("name", "Unnamed")
        expected = test.get("expected", "")
        criteria = test.get("criteria", [])
        
        # Handle multi-turn tests
        if test.get("multi_turn", False) and "steps" in test:
            # Use same session for all steps
            session_id = session_id or f"eval_{uuid.uuid4().hex[:8]}"
            
            responses = []
            for step in test["steps"]:
                response = self.client.chat_sync(
                    message=step,
                    user_id="eval_runner",
                    session_id=session_id
                )
                responses.append(f"User: {step}\nAI: {response.text}")
            
            actual_response = "\n\n".join(responses)
            input_text = " → ".join(test["steps"])
        else:
            # Single-turn test
            input_text = test.get("input", "")
            response = self.client.chat_sync(
                message=input_text,
                user_id="eval_runner",
                session_id=session_id
            )
            actual_response = response.text
        
        # Evaluate with LLM Judge
        eval_result = self.judge.evaluate(
            question=input_text,
            expected=expected,
            criteria=criteria,
            response=actual_response
        )
        
        duration = time.time() - start_time
        
        return TestResult(
            test_id=test_id,
            test_name=test_name,
            input=input_text,
            expected=expected,
            actual_response=actual_response[:500],  # Truncate for report
            score=eval_result.score,
            passed=eval_result.passed,
            reason=eval_result.reason,
            duration_seconds=round(duration, 2),
            criteria_met=eval_result.criteria_met
        )
    
    def run_set(self, set_name: str) -> SetResult:
        """Run all tests in a set"""
        # Find the set
        target_set = None
        for s in self.test_cases.get("sets", []):
            if s.get("name", "").lower() == set_name.lower():
                target_set = s
                break
        
        if not target_set:
            raise ValueError(f"Set '{set_name}' not found")
        
        results = []
        for test in target_set.get("tests", []):
            logger.info(f"Running test: {test.get('id')} - {test.get('name')}")
            result = self.run_single_test(test)
            results.append(result)
            
            status = "✅ PASS" if result.passed else "❌ FAIL"
            logger.info(f"  {status} (score: {result.score:.2f}) - {result.reason[:50]}")
        
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        avg_score = sum(r.score for r in results) / total if total > 0 else 0
        
        return SetResult(
            set_name=set_name,
            total_tests=total,
            passed_tests=passed,
            failed_tests=total - passed,
            pass_rate=round(passed / total * 100, 1) if total > 0 else 0,
            avg_score=round(avg_score, 2),
            results=results
        )
    
    def run_all(self) -> EvalReport:
        """Run all test sets"""
        set_results = []
        all_results = []
        
        for s in self.test_cases.get("sets", []):
            set_name = s.get("name", "Unknown")
            logger.info(f"\n{'='*60}")
            logger.info(f"Running Set: {set_name}")
            logger.info(f"{'='*60}")
            
            set_result = self.run_set(set_name)
            set_results.append(set_result)
            all_results.extend(set_result.results)
            
            logger.info(f"\nSet {set_name}: {set_result.passed_tests}/{set_result.total_tests} PASS ({set_result.pass_rate}%)")
        
        total = len(all_results)
        passed = sum(1 for r in all_results if r.passed)
        avg_score = sum(r.score for r in all_results) / total if total > 0 else 0
        
        return EvalReport(
            timestamp=datetime.now().isoformat(),
            total_tests=total,
            passed_tests=passed,
            failed_tests=total - passed,
            overall_pass_rate=round(passed / total * 100, 1) if total > 0 else 0,
            overall_avg_score=round(avg_score, 2),
            set_results=set_results
        )
    
    def save_report(self, report: EvalReport, output_dir: Optional[str] = None) -> str:
        """Save evaluation report to JSON"""
        output_dir = output_dir or str(Path(__file__).parent / "results")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_report_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"\nReport saved: {filepath}")
        return filepath
    
    def print_summary(self, report: EvalReport):
        """Print evaluation summary to console"""
        print("\n" + "="*60)
        print("📊 SCOOP AI EVALUATION REPORT")
        print("="*60)
        print(f"Timestamp: {report.timestamp}")
        print(f"\n{'Set':<15} {'Pass':<8} {'Fail':<8} {'Rate':<10} {'Avg Score'}")
        print("-"*60)
        
        for sr in report.set_results:
            print(f"{sr.set_name:<15} {sr.passed_tests:<8} {sr.failed_tests:<8} {sr.pass_rate}%{' '*5} {sr.avg_score}")
        
        print("-"*60)
        print(f"{'TOTAL':<15} {report.passed_tests:<8} {report.failed_tests:<8} {report.overall_pass_rate}%{' '*5} {report.overall_avg_score}")
        print("="*60)
        
        # Show failed tests
        failed_tests = [r for sr in report.set_results for r in sr.results if not r.passed]
        if failed_tests:
            print(f"\n❌ FAILED TESTS ({len(failed_tests)}):")
            for ft in failed_tests:
                print(f"  • [{ft.test_id}] {ft.test_name}: {ft.reason[:60]}")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Scoop AI Evaluation Runner")
    parser.add_argument("--set", "-s", help="Run specific set (Simple, Context, Medical, Ethics, Logic)")
    parser.add_argument("--test", "-t", help="Run specific test by ID (e.g., S1, M3)")
    parser.add_argument("--output", "-o", help="Output directory for reports")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    runner = EvalRunner()
    
    if args.test:
        # Run single test
        for s in runner.test_cases.get("sets", []):
            for test in s.get("tests", []):
                if test.get("id") == args.test:
                    result = runner.run_single_test(test)
                    print(f"\nTest: {result.test_id} - {result.test_name}")
                    print(f"Score: {result.score}")
                    print(f"Passed: {'✅' if result.passed else '❌'}")
                    print(f"Reason: {result.reason}")
                    return
        print(f"Test '{args.test}' not found")
        
    elif args.set:
        # Run specific set
        set_result = runner.run_set(args.set)
        print(f"\nSet: {set_result.set_name}")
        print(f"Pass Rate: {set_result.pass_rate}%")
        print(f"Avg Score: {set_result.avg_score}")
        
    else:
        # Run all
        report = runner.run_all()
        runner.print_summary(report)
        json_path = runner.save_report(report, args.output)
        
        # Generate HTML dashboard
        html_path = save_html_report(report.to_dict())
        logger.info(f"HTML Dashboard: {html_path}")
        
        # Auto-open in browser
        import webbrowser
        webbrowser.open(f"file://{html_path}")


if __name__ == "__main__":
    main()
