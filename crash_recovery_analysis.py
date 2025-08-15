#!/usr/bin/env python3
"""
🔍 CRASH RECOVERY ANALYSIS

After a PC crash, run this script to analyze what happened:
- Check the last saved progress files
- Identify the exact crash point
- Review memory usage patterns
- Generate crash summary
"""

import json
import glob
import os
from datetime import datetime

def analyze_crash_recovery():
    """Analyze crash recovery data"""
    print("🔍 CRASH RECOVERY ANALYSIS")
    print("=" * 60)
    
    # Find all progress files
    progress_files = sorted(glob.glob("/home/adra/JustNewsAgentic/crash_test_progress_*.json"))
    
    if not progress_files:
        print("❌ No progress files found")
        return
    
    print(f"📁 Found {len(progress_files)} progress files")
    
    # Get the last few progress files
    recent_files = progress_files[-5:] if len(progress_files) >= 5 else progress_files
    
    print(f"📊 Analyzing last {len(recent_files)} analyses:")
    
    memory_progression = []
    
    for i, file_path in enumerate(recent_files):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            analysis_num = data.get('analysis_count', i)
            image_name = data.get('image_file', 'unknown')
            success = data.get('success', False)
            
            # Memory data
            pre_mem = data.get('pre_analysis_state', {}).get('process_memory', {}).get('rss_mb', 0)
            post_mem = data.get('post_analysis_state', {}).get('process_memory', {}).get('rss_mb', 0)
            delta = data.get('memory_delta', {}).get('process_memory_change', 0)
            
            status = "✅" if success else "❌"
            print(f"  {status} Analysis {analysis_num}: {image_name}")
            print(f"     Memory: {pre_mem:.1f}MB → {post_mem:.1f}MB (Δ{delta:+.1f}MB)")
            
            memory_progression.append({
                'analysis': analysis_num,
                'memory_mb': post_mem,
                'delta': delta
            })
            
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
    
    # Check for crash report
    crash_report_path = "/home/adra/JustNewsAgentic/CRASH_REPORT.json"
    if os.path.exists(crash_report_path):
        print("\n💥 CRASH REPORT FOUND:")
        try:
            with open(crash_report_path, 'r') as f:
                crash_data = json.load(f)
            
            print(f"  Crash Point: Analysis {crash_data.get('crash_point', 'unknown')}")
            print(f"  Last Image: {crash_data.get('last_image', 'unknown')}")
            print(f"  Error: {crash_data.get('error', 'unknown')}")
            print(f"  Time: {crash_data.get('timestamp', 'unknown')}")
        except Exception as e:
            print(f"❌ Error reading crash report: {e}")
    
    # Memory analysis
    if len(memory_progression) >= 2:
        print(f"\n📈 MEMORY TREND:")
        start_mem = memory_progression[0]['memory_mb']
        end_mem = memory_progression[-1]['memory_mb']
        total_growth = end_mem - start_mem
        
        print(f"  Start: {start_mem:.1f}MB")
        print(f"  End: {end_mem:.1f}MB")
        print(f"  Total Growth: {total_growth:+.1f}MB")
        
        if total_growth > 500:  # 500MB growth
            print(f"  ⚠️ EXCESSIVE MEMORY GROWTH DETECTED!")
        
        # Check for memory leaks (consistent positive deltas)
        positive_deltas = [m['delta'] for m in memory_progression if m['delta'] > 10]
        if len(positive_deltas) >= 3:
            avg_leak = sum(positive_deltas) / len(positive_deltas)
            print(f"  🚨 POTENTIAL MEMORY LEAK: Avg +{avg_leak:.1f}MB per analysis")
    
    print("\n🔍 CRASH INVESTIGATION SUMMARY:")
    print(f"  - Last successful analysis: {progress_files[-1] if progress_files else 'none'}")
    print(f"  - Total analyses before crash: {len(progress_files)}")
    print(f"  - Check crash_test.log for detailed logs")

if __name__ == "__main__":
    analyze_crash_recovery()
