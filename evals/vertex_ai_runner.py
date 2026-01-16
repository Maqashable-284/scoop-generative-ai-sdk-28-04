"""
Scoop AI Evals - Vertex AI Gen AI Evaluation Service Runner
Uses Google Cloud's native evaluation framework for enterprise-grade assessment
"""
import os
import sys
import yaml
import pandas as pd
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Vertex AI imports
try:
    from vertexai import Client
    from vertexai import types
except ImportError:
    print("Installing Vertex AI SDK...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-cloud-aiplatform[evaluation]"])
    from vertexai import Client
    from vertexai import types

from evals.client import create_client as create_scoop_client


# Configuration
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "scoop-generative-ai-sdk-28-04")
LOCATION = os.getenv("VERTEX_AI_LOCATION", "europe-west1")


def load_test_cases() -> list:
    """Load test cases from YAML"""
    yaml_path = Path(__file__).parent / "test_cases.yaml"
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    test_cases = []
    for test_set in data.get("sets", []):
        set_name = test_set.get("name", "Unknown")
        for test in test_set.get("tests", []):
            test_cases.append({
                "id": test.get("id"),
                "name": test.get("name"),
                "set_name": set_name,
                "input": test.get("input", ""),
                "expected": test.get("expected", ""),
                "criteria": test.get("criteria", []),
                "multi_turn": test.get("multi_turn", False),
                "steps": test.get("steps", [])
            })
    return test_cases


def generate_scoop_responses(test_cases: list) -> pd.DataFrame:
    """
    Generate responses from Scoop AI backend (not directly from Gemini model).
    This evaluates the entire system, not just the model.
    """
    print("🔄 Generating Scoop AI responses...")
    scoop_client = create_scoop_client()
    
    results = []
    for i, tc in enumerate(test_cases):
        print(f"  [{i+1}/{len(test_cases)}] {tc['id']}: {tc['name'][:40]}...")
        
        try:
            if tc["multi_turn"] and tc["steps"]:
                # Handle multi-turn conversation
                session_id = f"vertex_eval_{tc['id']}"
                responses = []
                for step in tc["steps"]:
                    response = scoop_client.chat_sync(
                        message=step,
                        user_id="vertex_eval",
                        session_id=session_id
                    )
                    responses.append(f"User: {step}\nAI: {response.text}")
                response_text = "\n\n".join(responses)
                prompt_text = " → ".join(tc["steps"])
            else:
                # Single-turn
                response = scoop_client.chat_sync(
                    message=tc["input"],
                    user_id="vertex_eval"
                )
                response_text = response.text
                prompt_text = tc["input"]
            
            results.append({
                "test_id": tc["id"],
                "test_name": tc["name"],
                "set_name": tc["set_name"],
                "prompt": prompt_text,
                "reference": tc["expected"],
                "response": response_text,
                "criteria": "; ".join(tc["criteria"]) if tc["criteria"] else ""
            })
        except Exception as e:
            print(f"    ❌ Error: {e}")
            results.append({
                "test_id": tc["id"],
                "test_name": tc["name"],
                "set_name": tc["set_name"],
                "prompt": tc.get("input", " → ".join(tc.get("steps", []))),
                "reference": tc["expected"],
                "response": f"[ERROR] {str(e)}",
                "criteria": "; ".join(tc["criteria"]) if tc["criteria"] else ""
            })
    
    return pd.DataFrame(results)


def run_vertex_evaluation(
    eval_df: pd.DataFrame,
    metric_type: str = "general_quality"
) -> dict:
    """
    Run Vertex AI Gen AI Evaluation on pre-generated responses.
    """
    print(f"\n📊 Running Vertex AI Evaluation with {metric_type}...")
    
    # Initialize Vertex AI client
    client = Client(project=PROJECT_ID, location=LOCATION)
    
    # Prepare dataset for Vertex AI evaluation
    # The SDK expects 'prompt' and 'response' columns
    vertex_df = eval_df[["prompt", "response", "reference"]].copy()
    
    # Choose metric based on type
    if metric_type == "general_quality":
        metrics = [types.RubricMetric.GENERAL_QUALITY]
    elif metric_type == "custom":
        # Custom rubric for Scoop AI specific evaluation
        metrics = [types.CustomMetric(
            name="scoop_quality",
            definition="Evaluate if the response correctly addresses a sports nutrition query in Georgian",
            rubric={
                1: "Response is incorrect, irrelevant, or contains errors",
                2: "Response is partially correct but missing key information",
                3: "Response is correct but could be more helpful",
                4: "Response is correct, helpful, and well-formatted",
                5: "Response is excellent, comprehensive, and shows expertise"
            }
        )]
    else:
        metrics = [types.RubricMetric.GENERAL_QUALITY]
    
    # Run evaluation
    eval_result = client.evals.evaluate(
        dataset=vertex_df,
        metrics=metrics
    )
    
    # Extract and print results for terminal
    try:
        results_df = eval_result.to_dataframe()
        print("\n" + "="*60)
        print("📊 VERTEX AI EVALUATION RESULTS")
        print("="*60)
        
        # Calculate summary stats
        if 'general_quality_v1/score' in results_df.columns:
            scores = results_df['general_quality_v1/score'].dropna()
            avg_score = scores.mean() if len(scores) > 0 else 0
            pass_count = (scores >= 0.7).sum()
            fail_count = (scores < 0.7).sum()
            
            print(f"\n📈 Summary:")
            print(f"   Average Score: {avg_score:.2f}")
            print(f"   Passed (≥0.7): {pass_count}/{len(scores)} ({100*pass_count/len(scores):.1f}%)")
            print(f"   Failed (<0.7): {fail_count}/{len(scores)}")
            
            # Show individual results
            print(f"\n📝 Individual Results:")
            print("-"*60)
            for idx, row in results_df.iterrows():
                score = row.get('general_quality_v1/score', 'N/A')
                verdict = row.get('general_quality_v1/verdict', 'N/A')
                prompt = str(row.get('prompt', ''))[:50]
                status = "✅" if score >= 0.7 else "❌"
                print(f"  {status} [{idx+1:02d}] Score: {score:.2f} | {prompt}...")
        
        # Save to JSON
        output_path = Path(__file__).parent / "results" / "vertex_ai_results.json"
        output_path.parent.mkdir(exist_ok=True)
        results_df.to_json(output_path, orient='records', force_ascii=False, indent=2)
        print(f"\n💾 Results saved to: {output_path}")
        
    except Exception as e:
        print(f"⚠️ Could not extract results: {e}")
    
    return {
        "result": eval_result,
        "df": eval_df
    }


def run_full_evaluation(
    set_filter: Optional[str] = None,
    use_vertex_api: bool = True
):
    """
    Main entry point: Generate Scoop responses and evaluate with Vertex AI
    """
    # Load test cases
    test_cases = load_test_cases()
    
    if set_filter:
        test_cases = [tc for tc in test_cases if tc["set_name"].lower() == set_filter.lower()]
        print(f"📋 Filtered to {len(test_cases)} tests in '{set_filter}' set")
    else:
        print(f"📋 Running all {len(test_cases)} tests")
    
    # Generate Scoop AI responses
    eval_df = generate_scoop_responses(test_cases)
    
    print(f"\n✅ Generated {len(eval_df)} responses")
    
    if use_vertex_api:
        # Run Vertex AI evaluation
        try:
            result = run_vertex_evaluation(eval_df)
            print("\n📈 Vertex AI Evaluation Complete!")
            result["result"].show()
            return result
        except Exception as e:
            print(f"\n⚠️ Vertex AI evaluation failed: {e}")
            print("Falling back to local evaluation results...")
    
    # Return DataFrame for local analysis
    return {
        "df": eval_df,
        "summary": {
            "total": len(eval_df),
            "by_set": eval_df.groupby("set_name").size().to_dict()
        }
    }


def quick_test():
    """
    Quick test without full Vertex AI integration.
    Generates responses and displays them locally.
    """
    print("🚀 Quick Vertex AI Eval Test Mode\n")
    
    # Load just 3 test cases
    test_cases = load_test_cases()[:3]
    
    # Generate responses
    eval_df = generate_scoop_responses(test_cases)
    
    print("\n" + "="*60)
    print("📝 Generated Responses:")
    print("="*60)
    
    for _, row in eval_df.iterrows():
        print(f"\n[{row['test_id']}] {row['test_name']}")
        print(f"Input: {row['prompt'][:100]}...")
        print(f"Expected: {row['reference']}")
        print(f"Response: {row['response'][:200]}...")
        print("-"*40)
    
    return eval_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scoop AI Vertex AI Evaluation Runner")
    parser.add_argument("--set", "-s", help="Run specific set only")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick test mode (no Vertex API)")
    parser.add_argument("--project", "-p", help="GCP Project ID")
    parser.add_argument("--location", "-l", default="europe-west1", help="Vertex AI location")
    
    args = parser.parse_args()
    
    if args.project:
        PROJECT_ID = args.project
    
    if args.quick:
        quick_test()
    else:
        run_full_evaluation(set_filter=args.set, use_vertex_api=True)
