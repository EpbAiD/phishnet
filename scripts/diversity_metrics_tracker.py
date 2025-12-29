#!/usr/bin/env python3
"""
Diversity Metrics Tracker
==========================
Tracks and monitors diversity across 8 dimensions:
1. Protocol diversity
2. Category diversity
3. TLD diversity
4. Geographic diversity
5. Structural diversity
6. Attack type diversity
7. Temporal diversity
8. Traffic tier diversity

Calculates coverage scores and identifies gaps.
"""

import pandas as pd
import numpy as np
from scipy.stats import entropy
from collections import Counter
from datetime import datetime, timedelta
import json


class DiversityMetricsTracker:
    """Track diversity metrics across all dimensions"""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with URL dataset

        Args:
            df: DataFrame with columns: url, label, source, collected_date, etc.
        """
        self.df = df
        self.metrics = {}

    def calculate_protocol_diversity(self) -> dict:
        """Calculate protocol diversity metrics"""
        # Extract protocol from URL
        self.df['protocol'] = self.df['url'].str.extract(r'^([a-z-]+)://')[0]
        self.df['protocol'] = self.df['protocol'].fillna('http')  # Default

        protocol_counts = self.df['protocol'].value_counts()

        # Calculate entropy
        protocol_entropy = entropy(protocol_counts.values, base=2)

        # Coverage score (out of 30 expected protocols)
        unique_protocols = self.df['protocol'].nunique()
        protocol_coverage = min(unique_protocols / 30.0, 1.0) * 100

        return {
            'unique_protocols': int(unique_protocols),
            'protocol_entropy': float(protocol_entropy),
            'protocol_coverage_score': float(protocol_coverage),
            'top_protocols': protocol_counts.head(10).to_dict()
        }

    def calculate_category_diversity(self) -> dict:
        """Calculate category diversity metrics"""
        if 'category' not in self.df.columns:
            return {'error': 'category column missing'}

        category_counts = self.df['category'].value_counts()

        # Calculate entropy
        category_entropy = entropy(category_counts.values, base=2)

        # Coverage score (out of 50 expected categories)
        unique_categories = self.df['category'].nunique()
        category_coverage = min(unique_categories / 50.0, 1.0) * 100

        return {
            'unique_categories': int(unique_categories),
            'category_entropy': float(category_entropy),
            'category_coverage_score': float(category_coverage),
            'top_categories': category_counts.head(15).to_dict()
        }

    def calculate_tld_diversity(self) -> dict:
        """Calculate TLD diversity metrics"""
        # Extract TLD from URL
        self.df['tld'] = self.df['url'].str.extract(r'\.([a-z]{2,})(?:/|:|$)')[0]

        tld_counts = self.df['tld'].dropna().value_counts()

        # Calculate entropy
        tld_entropy = entropy(tld_counts.values, base=2)

        # Coverage score (out of 200 target TLDs)
        unique_tlds = self.df['tld'].nunique()
        tld_coverage = min(unique_tlds / 200.0, 1.0) * 100

        return {
            'unique_tlds': int(unique_tlds),
            'tld_entropy': float(tld_entropy),
            'tld_coverage_score': float(tld_coverage),
            'top_tlds': tld_counts.head(20).to_dict()
        }

    def calculate_geographic_diversity(self) -> dict:
        """Calculate geographic diversity metrics"""
        if 'region' not in self.df.columns:
            return {'error': 'region column missing'}

        region_counts = self.df['region'].value_counts()

        # Calculate entropy
        region_entropy = entropy(region_counts.values, base=2)

        # Coverage score (out of 10 expected regions)
        unique_regions = self.df['region'].nunique()
        region_coverage = min(unique_regions / 10.0, 1.0) * 100

        return {
            'unique_regions': int(unique_regions),
            'region_entropy': float(region_entropy),
            'region_coverage_score': float(region_coverage),
            'region_distribution': region_counts.to_dict()
        }

    def calculate_structural_diversity(self) -> dict:
        """Calculate structural diversity metrics"""
        # URL length variance
        url_lengths = self.df['url'].str.len()
        length_variance = url_lengths.var()
        length_std = url_lengths.std()

        # Length buckets
        length_buckets = pd.cut(url_lengths, bins=[0, 20, 50, 100, 200, 1000])
        length_distribution = length_buckets.value_counts().to_dict()

        # Structural features
        has_port = self.df['url'].str.contains(':[0-9]{2,5}/', regex=True).sum()
        has_query = self.df['url'].str.contains('\?').sum()
        has_fragment = self.df['url'].str.contains('#').sum()

        return {
            'url_length_mean': float(url_lengths.mean()),
            'url_length_std': float(length_std),
            'url_length_variance': float(length_variance),
            'length_distribution': {str(k): int(v) for k, v in length_distribution.items()},
            'urls_with_port': int(has_port),
            'urls_with_query': int(has_query),
            'urls_with_fragment': int(has_fragment),
            'structural_coverage_score': float(min((length_std / 50.0), 1.0) * 100)
        }

    def calculate_attack_diversity(self) -> dict:
        """Calculate attack type diversity (phishing only)"""
        phish_df = self.df[self.df['label'] == 'phishing']

        if 'attack_type' not in phish_df.columns:
            return {'error': 'attack_type column missing'}

        attack_counts = phish_df['attack_type'].value_counts()

        # Calculate entropy
        attack_entropy = entropy(attack_counts.values, base=2)

        # Coverage score (out of 10 expected attack types)
        unique_attacks = phish_df['attack_type'].nunique()
        attack_coverage = min(unique_attacks / 10.0, 1.0) * 100

        return {
            'unique_attack_types': int(unique_attacks),
            'attack_entropy': float(attack_entropy),
            'attack_coverage_score': float(attack_coverage),
            'attack_distribution': attack_counts.to_dict()
        }

    def calculate_temporal_diversity(self) -> dict:
        """Calculate temporal diversity metrics"""
        if 'collected_date' not in self.df.columns:
            return {'error': 'collected_date column missing'}

        # Convert to datetime
        self.df['collected_datetime'] = pd.to_datetime(self.df['collected_date'])

        # Calculate freshness (% collected in last 30 days)
        now = datetime.now()
        thirty_days_ago = now - timedelta(days=30)

        fresh_count = (self.df['collected_datetime'] > thirty_days_ago).sum()
        freshness_pct = (fresh_count / len(self.df)) * 100

        # Coverage score (target 30% fresh data)
        temporal_coverage = min(freshness_pct / 30.0, 1.0) * 100

        return {
            'total_urls': len(self.df),
            'fresh_urls_30d': int(fresh_count),
            'freshness_pct': float(freshness_pct),
            'temporal_coverage_score': float(temporal_coverage)
        }

    def calculate_traffic_diversity(self) -> dict:
        """Calculate traffic tier diversity (top sites vs long-tail)"""
        # Extract domain
        self.df['domain'] = self.df['url'].str.extract(r'://(?:www\.)?([^/:]+)')[0]

        # Calculate Gini coefficient (measure of inequality)
        domain_counts = self.df['domain'].value_counts().values
        sorted_counts = np.sort(domain_counts)
        n = len(sorted_counts)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n

        # Lower Gini = more diverse (target 0.5)
        # Higher Gini = concentrated (e.g., 0.9 = top sites only)
        traffic_coverage = (1 - min(gini / 0.5, 1.0)) * 100

        return {
            'unique_domains': int(self.df['domain'].nunique()),
            'total_urls': len(self.df),
            'avg_urls_per_domain': float(len(self.df) / self.df['domain'].nunique()),
            'gini_coefficient': float(gini),
            'traffic_coverage_score': float(traffic_coverage),
            'top_10_domains': self.df['domain'].value_counts().head(10).to_dict()
        }

    def calculate_overall_score(self) -> float:
        """Calculate overall diversity coverage score (0-100)"""
        weights = {
            'protocol': 0.15,
            'category': 0.20,
            'tld': 0.10,
            'geographic': 0.10,
            'structural': 0.10,
            'attack': 0.15,
            'temporal': 0.10,
            'traffic': 0.10
        }

        overall_score = 0.0

        for dimension, weight in weights.items():
            if dimension in self.metrics:
                score = self.metrics[dimension].get(f'{dimension}_coverage_score', 0)
                overall_score += score * weight

        return float(overall_score)

    def analyze(self) -> dict:
        """Run complete diversity analysis"""
        print("=" * 80)
        print("DIVERSITY METRICS ANALYSIS")
        print("=" * 80)
        print(f"Dataset: {len(self.df):,} URLs")
        print("")

        # Calculate all dimensions
        self.metrics = {
            'protocol': self.calculate_protocol_diversity(),
            'category': self.calculate_category_diversity(),
            'tld': self.calculate_tld_diversity(),
            'geographic': self.calculate_geographic_diversity(),
            'structural': self.calculate_structural_diversity(),
            'attack': self.calculate_attack_diversity(),
            'temporal': self.calculate_temporal_diversity(),
            'traffic': self.calculate_traffic_diversity()
        }

        # Calculate overall score
        overall_score = self.calculate_overall_score()
        self.metrics['overall'] = {
            'diversity_score': overall_score,
            'grade': self.get_grade(overall_score)
        }

        # Print summary
        print("Dimension Coverage Scores:")
        print("-" * 80)
        for dimension, metrics in self.metrics.items():
            if dimension == 'overall':
                continue
            score = metrics.get(f'{dimension}_coverage_score', 0)
            status = "✅" if score >= 70 else "⚠️" if score >= 40 else "❌"
            print(f"  {status} {dimension.capitalize():<15}: {score:>6.1f}%")

        print("")
        print("=" * 80)
        print(f"OVERALL DIVERSITY SCORE: {overall_score:.1f}/100 ({self.metrics['overall']['grade']})")
        print("=" * 80)

        return self.metrics

    def get_grade(self, score: float) -> str:
        """Get letter grade for diversity score"""
        if score >= 85:
            return "A (Excellent)"
        elif score >= 70:
            return "B (Good)"
        elif score >= 55:
            return "C (Fair)"
        elif score >= 40:
            return "D (Poor)"
        else:
            return "F (Critical)"

    def save_report(self, output_path: str):
        """Save metrics report to JSON"""
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"\n✅ Metrics saved to: {output_path}")


# ============================================================================
# CLI Entry Point
# ============================================================================
def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Diversity Metrics Tracker')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, default='logs/diversity_metrics.json', help='Output JSON report')

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)

    # Analyze
    tracker = DiversityMetricsTracker(df)
    metrics = tracker.analyze()

    # Save report
    tracker.save_report(args.output)


if __name__ == '__main__':
    main()
