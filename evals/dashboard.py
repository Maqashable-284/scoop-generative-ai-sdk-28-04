"""
Scoop AI Evals - HTML Dashboard Generator
Generates a pretty HTML report from evaluation results
"""
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def generate_html(report: Dict[str, Any]) -> str:
    """Generate HTML dashboard from report data"""
    
    timestamp = report.get("timestamp", datetime.now().isoformat())
    total = report.get("total_tests", 0)
    passed = report.get("passed_tests", 0)
    failed = report.get("failed_tests", 0)
    pass_rate = report.get("overall_pass_rate", 0)
    avg_score = report.get("overall_avg_score", 0)
    
    # Generate set sections
    set_sections = ""
    for sr in report.get("set_results", []):
        set_name = sr.get("set_name", "Unknown")
        set_passed = sr.get("passed_tests", 0)
        set_failed = sr.get("failed_tests", 0)
        
        # Generate test rows
        test_rows = ""
        for r in sr.get("results", []):
            status_class = "badge-pass" if r.get("passed") else "badge-fail"
            status_text = "PASS" if r.get("passed") else "FAIL"
            score = r.get("score", 0)
            score_percent = int(score * 100)
            
            test_rows += f"""
<tr>
    <td><strong>{r.get('test_id', 'N/A')}</strong></td>
    <td>{r.get('test_name', 'Unknown')}</td>
    <td><span class="badge {status_class}">{status_text}</span></td>
    <td>
        <div class="score-bar"><div class="score-bar-fill" style="width: {score_percent}%"></div></div>
        {score:.2f}
    </td>
    <td class="reason">{r.get('reason', '')[:80]}</td>
</tr>
<tr class="response-row">
    <td colspan="5">
        <details>
            <summary style="cursor:pointer;color:#3b82f6;font-weight:600;">📝 კითხვა და პასუხი</summary>
            <div style="margin-top:0.5rem;padding:1rem;background:rgba(15,23,42,0.8);border-radius:8px;">
                <div style="color:#22c55e;margin-bottom:0.5rem;"><strong>📨 Input:</strong> {r.get('input', '')}</div>
                <div style="color:#94a3b8;margin-bottom:0.5rem;"><strong>🎯 Expected:</strong> {r.get('expected', '')}</div>
                <div style="color:#e2e8f0;white-space:pre-wrap;word-break:break-word;"><strong>🤖 AI Response:</strong><br>{r.get('actual_response', '')}</div>
            </div>
        </details>
    </td>
</tr>
"""
        
        set_sections += f"""
<div class="set-section">
    <div class="set-header">
        <h2>{set_name}</h2>
        <div class="set-stats">
            <span class="stat-pass">✅ {set_passed} Pass</span>
            <span class="stat-fail">❌ {set_failed} Fail</span>
        </div>
    </div>
    <table class="tests-table">
        <thead>
            <tr>
                <th>ID</th>
                <th>Test Name</th>
                <th>Status</th>
                <th>Score</th>
                <th>Reason</th>
            </tr>
        </thead>
        <tbody>
            {test_rows}
        </tbody>
    </table>
</div>
"""
    
    html = f"""<!DOCTYPE html>
<html lang="ka">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scoop AI Evals Dashboard</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #e2e8f0;
            min-height: 100vh;
            padding: 2rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ text-align: center; margin-bottom: 2rem; }}
        .header h1 {{
            font-size: 2.5rem;
            background: linear-gradient(90deg, #22c55e, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        .header .timestamp {{ color: #64748b; font-size: 0.9rem; }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .card {{
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
        }}
        .card .value {{ font-size: 2.5rem; font-weight: 700; }}
        .card .label {{ color: #94a3b8; margin-top: 0.5rem; }}
        .card.pass .value {{ color: #22c55e; }}
        .card.fail .value {{ color: #ef4444; }}
        .card.score .value {{ color: #3b82f6; }}
        .progress-bar {{
            background: #1e293b;
            border-radius: 8px;
            height: 24px;
            overflow: hidden;
            margin: 1rem 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #22c55e, #16a34a);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 0.85rem;
        }}
        .set-section {{
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid #334155;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            overflow: hidden;
        }}
        .set-header {{
            background: rgba(51, 65, 85, 0.5);
            padding: 1rem 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .set-header h2 {{ font-size: 1.25rem; }}
        .set-stats {{ display: flex; gap: 1rem; }}
        .set-stats span {{
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .stat-pass {{ background: rgba(34, 197, 94, 0.2); color: #22c55e; }}
        .stat-fail {{ background: rgba(239, 68, 68, 0.2); color: #ef4444; }}
        .tests-table {{ width: 100%; border-collapse: collapse; }}
        .tests-table th {{
            background: rgba(51, 65, 85, 0.3);
            padding: 0.75rem 1rem;
            text-align: left;
            font-weight: 600;
            color: #94a3b8;
        }}
        .tests-table td {{
            padding: 0.75rem 1rem;
            border-top: 1px solid #334155;
        }}
        .tests-table tr:hover {{ background: rgba(51, 65, 85, 0.3); }}
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.8rem;
        }}
        .badge-pass {{ background: #22c55e; color: #052e16; }}
        .badge-fail {{ background: #ef4444; color: #450a0a; }}
        .score-bar {{
            width: 60px;
            height: 8px;
            background: #334155;
            border-radius: 4px;
            overflow: hidden;
            display: inline-block;
            margin-right: 0.5rem;
        }}
        .score-bar-fill {{ height: 100%; background: #3b82f6; }}
        .reason {{ font-size: 0.85rem; color: #94a3b8; max-width: 300px; }}
        .footer {{
            text-align: center;
            color: #64748b;
            padding: 2rem 0;
            font-size: 0.85rem;
        }}
        .braintrust-link {{
            display: inline-block;
            margin-top: 1rem;
            padding: 0.75rem 1.5rem;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>🧪 Scoop AI Evals Dashboard</h1>
            <p class="timestamp">Generated: {timestamp}</p>
            <a href="https://braintrust.dev/app" target="_blank" class="braintrust-link">
                📊 View in Braintrust Dashboard →
            </a>
        </header>
        
        <div class="summary">
            <div class="card pass">
                <div class="value">{passed}</div>
                <div class="label">Passed Tests</div>
            </div>
            <div class="card fail">
                <div class="value">{failed}</div>
                <div class="label">Failed Tests</div>
            </div>
            <div class="card score">
                <div class="value">{pass_rate}%</div>
                <div class="label">Pass Rate</div>
            </div>
            <div class="card score">
                <div class="value">{avg_score:.2f}</div>
                <div class="label">Avg Score</div>
            </div>
        </div>
        
        <div class="progress-bar">
            <div class="progress-fill" style="width: {pass_rate}%">
                {passed}/{total} Tests Passed
            </div>
        </div>
        
        {set_sections}
        
        <footer class="footer">
            <p>Scoop AI Evaluation Framework • Powered by Braintrust & Gemini</p>
        </footer>
    </div>
</body>
</html>"""
    
    return html


def save_html_report(report: Dict[str, Any], output_dir: str = None) -> str:
    """Save HTML report to file"""
    output_dir = output_dir or str(Path(__file__).parent / "results")
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, "dashboard.html")
    html = generate_html(report)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return filepath
