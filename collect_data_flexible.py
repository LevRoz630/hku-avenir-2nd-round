#!/usr/bin/env python3
"""
Flexible data collection script with configurable timeframes and periods
"""

import argparse
import sys
from data_collector import DataCollector

def main():
    parser = argparse.ArgumentParser(description='Flexible Data Collection for Trading Strategy')
    parser.add_argument('--timeframe', choices=['1h', '15m', '1m'], default='15m',
                       help='Data granularity: 1h (hourly), 15m (15-minute), 1m (1-minute)')
    parser.add_argument('--days', type=int, default=7,
                       help='Number of days to collect (default: 7)')
    parser.add_argument('--mode', choices=['test', 'prod'], default='test',
                       help='OMS mode: test or prod (default: test)')
    parser.add_argument('--output-dir', default='test_data',
                       help='Output directory for data files (default: test_data)')
    
    args = parser.parse_args()
    
    print("=== Data Collection ===")
    print(f"Timeframe: {args.timeframe}")
    print(f"Period: {args.days} days")
    print(f"Mode: {args.mode}")
    
    # Initialize collector
    collector = DataCollector(args.output_dir, mode=args.mode)
    
    try:
        # Collect historical data
        print(f"\n=== Collecting {args.timeframe} data for {args.days} days ===")
        historical = collector.collect_historical_data(
            days=args.days, 
            timeframe=args.timeframe
        )
        
        
        # Create summary
        summary = {
            'collection_time': collector.data_dir.name,
            'symbols_collected': list(historical.keys()),
            'total_records': sum(len(df) for df in historical.values()),
            'account_available': collector.oms_client is not None,
            'data_directory': str(collector.data_dir),
            'timeframe': args.timeframe,
            'days': args.days
        }
        
        print(f"\nCollection complete: {summary['total_records']} records saved to {summary['data_directory']}")
            
    except Exception as e:
        print(f"Error during data collection: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
