"""
Scoop AI Evals - Braintrust Integration
=======================================

Runs 25 test cases against Scoop AI backend and logs results to Braintrust.

Usage:
    python -m evals.braintrust_runner                  # Run all tests
    python -m evals.braintrust_runner --set Simple     # Run specific set
    python -m evals.braintrust_runner --test S1        # Run single test

Requirements:
    pip install braintrust autoevals
    
Environment:
    BRAINTRUST_API_KEY=<your-api-key>
    BACKEND_URL=http://localhost:8080 (optional)
"""
import os
import sys
import yaml
import uuid
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Braintrust imports
from braintrust import Eval, init_logger, traced, Score

# Local imports
from evals.client import ScoopClient, create_client
from evals.judge import LLMJudge, create_judge

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Braintrust logger
BRAINTRUST_PROJECT = os.getenv("BRAINTRUST_PROJECT", "Scoop AI Evals")


@dataclass
class TestCase:
    """Single test case from YAML"""
    id: str
    name: str
    input: str
    expected: str
    criteria: List[str]
    multi_turn: bool = False
    steps: Optional[List[str]] = None
    severity: Optional[str] = None
    set_name: str = "Unknown"


def load_test_cases() -> List[TestCase]:
    """Load all test cases from YAML file"""
    yaml_path = Path(__file__).parent / "test_cases.yaml"
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    test_cases = []
    for test_set in data.get("sets", []):
        set_name = test_set.get("name", "Unknown")
        for test in test_set.get("tests", []):
            # Handle multi-turn tests
            if test.get("multi_turn", False) and "steps" in test:
                input_text = " → ".join(test["steps"])
            else:
                input_text = test.get("input", "")
            
            test_cases.append(TestCase(
                id=test.get("id", "unknown"),
                name=test.get("name", "Unnamed"),
                input=input_text,
                expected=test.get("expected", ""),
                criteria=test.get("criteria", []),
                multi_turn=test.get("multi_turn", False),
                steps=test.get("steps"),
                severity=test.get("severity"),
                set_name=set_name
            ))
    
    return test_cases


def create_scoop_task(client: ScoopClient, judge: LLMJudge):
    """Create the task function for Braintrust Eval"""
    
    @traced
    def task(input, hooks=None) -> str:
        """Execute a single test against Scoop AI backend"""
        # Input can be string (question) or dict (with test data)
        if isinstance(input, dict):
            # Handle multi-turn from dict
            if input.get("multi_turn") and input.get("steps"):
                steps = input["steps"]
                session_id = f"eval_{uuid.uuid4().hex[:8]}"
                responses = []
                for step in steps:
                    response = client.chat_sync(
                        message=step,
                        user_id="braintrust_eval",
                        session_id=session_id
                    )
                    responses.append(f"User: {step}\nAI: {response.text}")
                return "\n\n".join(responses)
            else:
                # Single turn from dict
                question = input.get("question", str(input))
                response = client.chat_sync(
                    message=question,
                    user_id="braintrust_eval"
                )
                return response.text
        else:
            # Simple string input
            response = client.chat_sync(
                message=str(input),
                user_id="braintrust_eval"
            )
            return response.text
    
    return task


def llm_judge_scorer(judge: LLMJudge):
    """Create an LLM Judge scorer for Braintrust"""
    
    def scorer(input, output: str, expected: str, metadata: Dict = None) -> Score:
        """Score the response using our existing LLM Judge"""
        # Extract question from input (could be dict or string)
        if isinstance(input, dict):
            question = input.get("question", " → ".join(input.get("steps", [])) if input.get("multi_turn") else str(input))
            criteria = input.get("criteria", [])
        else:
            question = str(input)
            criteria = metadata.get("criteria", []) if metadata else []
        
        # Use our existing judge
        result = judge.evaluate(
            question=question,
            expected=expected,
            criteria=criteria,
            response=output
        )
        
        # Return Braintrust Score object
        return Score(
            name="LLM Judge",
            score=result.score,
            metadata={
                "passed": result.passed,
                "reason": result.reason,
                "criteria_met": result.criteria_met
            }
        )
    
    return scorer


def run_braintrust_eval(
    test_cases: List[TestCase],
    set_filter: Optional[str] = None,
    test_filter: Optional[str] = None
):
    """Run evaluation using Braintrust"""
    
    # Filter test cases
    if test_filter:
        test_cases = [tc for tc in test_cases if tc.id == test_filter]
    elif set_filter:
        test_cases = [tc for tc in test_cases if tc.set_name.lower() == set_filter.lower()]
    
    if not test_cases:
        logger.error("No test cases found matching filter")
        return
    
    logger.info(f"🚀 Running {len(test_cases)} tests with Braintrust...")
    
    # Create clients
    client = create_client()
    judge = create_judge()
    
    # Convert test cases to Braintrust format
    def data_generator():
        for tc in test_cases:
            # For multi-turn, pass full test details so task can handle steps
            if tc.multi_turn and tc.steps:
                test_input = {
                    "multi_turn": True,
                    "steps": tc.steps,
                    "question": " → ".join(tc.steps),
                    "criteria": tc.criteria
                }
            else:
                test_input = {
                    "multi_turn": False,
                    "question": tc.input,
                    "criteria": tc.criteria
                }
            
            yield {
                "input": test_input,
                "expected": tc.expected,
                "metadata": {
                    "id": tc.id,
                    "name": tc.name,
                    "severity": tc.severity,
                    "set_name": tc.set_name
                }
            }
    
    # Determine experiment name based on filter
    if test_filter:
        experiment_name = f"Single Test: {test_filter}"
    elif set_filter:
        experiment_name = f"Set: {set_filter}"
    else:
        experiment_name = "Full Suite (25 tests)"
    
    # Run the evaluation
    Eval(
        name=BRAINTRUST_PROJECT,
        experiment_name=experiment_name,
        data=data_generator,
        task=create_scoop_task(client, judge),
        scores=[llm_judge_scorer(judge)],
        metadata={
            "backend_url": client.base_url,
            "test_count": len(test_cases),
            "filter_set": set_filter,
            "filter_test": test_filter
        }
    )
    
    logger.info("✅ Evaluation complete! View results at: https://braintrust.dev/app")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Scoop AI Braintrust Evaluation")
    parser.add_argument("--set", "-s", help="Run specific set (Simple, Context, Medical, Ethics, Logic)")
    parser.add_argument("--test", "-t", help="Run specific test by ID (e.g., S1, M3)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check for API key
    if not os.getenv("BRAINTRUST_API_KEY"):
        logger.error("❌ BRAINTRUST_API_KEY environment variable not set!")
        logger.info("   Get your API key at: https://braintrust.dev/app/settings?subroute=api-keys")
        sys.exit(1)
    
    # Load test cases
    test_cases = load_test_cases()
    logger.info(f"📋 Loaded {len(test_cases)} test cases from YAML")
    
    # Run evaluation
    run_braintrust_eval(
        test_cases=test_cases,
        set_filter=args.set,
        test_filter=args.test
    )


if __name__ == "__main__":
    main()
