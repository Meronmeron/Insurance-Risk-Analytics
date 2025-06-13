import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskAnalysis:
    def __init__(self, data_path: str):
        """Initialize the risk analysis with data loading and preprocessing."""
        self.data = pd.read_csv(data_path)
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocess the data for analysis."""
        # Calculate key metrics
        self.data['ClaimFrequency'] = (self.data['TotalClaims'] > 0).astype(int)
        self.data['ClaimSeverity'] = self.data['TotalClaims'] / self.data['TotalPolicies']
        self.data['Margin'] = self.data['TotalPremium'] - self.data['TotalClaims']
        
        logger.info("Data preprocessing completed")
        
    def test_province_risk_differences(self) -> Tuple[float, bool, float]:
        """
        Test H₀: There are no risk differences across provinces
        Using ANOVA for multiple group comparison
        """
        # Group data by province
        province_groups = [group for _, group in self.data.groupby('Province')]
        
        # Perform ANOVA test on LossRatio
        f_stat, p_value = stats.f_oneway(*[group['LossRatio'] for group in province_groups])
        
        # Calculate effect size (Eta-squared)
        grand_mean = self.data['LossRatio'].mean()
        total_ss = sum((self.data['LossRatio'] - grand_mean)**2)
        eta_squared = f_stat / (f_stat + len(self.data) - len(province_groups))
        
        return p_value, p_value < 0.05, eta_squared
    
    def test_vehicle_type_risk_differences(self) -> Tuple[float, bool, float]:
        """
        Test H₀: There are no risk differences between vehicle types
        Using ANOVA for multiple group comparison
        """
        # Group data by vehicle type
        vehicle_groups = [group for _, group in self.data.groupby('VehicleType')]
        
        # Perform ANOVA test on LossRatio
        f_stat, p_value = stats.f_oneway(*[group['LossRatio'] for group in vehicle_groups])
        
        # Calculate effect size
        grand_mean = self.data['LossRatio'].mean()
        total_ss = sum((self.data['LossRatio'] - grand_mean)**2)
        eta_squared = f_stat / (f_stat + len(self.data) - len(vehicle_groups))
        
        return p_value, p_value < 0.05, eta_squared
    
    def test_risk_category_differences(self) -> Tuple[float, bool, float]:
        """
        Test H₀: There are no significant differences between risk categories
        Using ANOVA for multiple group comparison
        """
        # Group data by risk category
        risk_groups = [group for _, group in self.data.groupby('RiskCategory')]
        
        # Perform ANOVA test on LossRatio
        f_stat, p_value = stats.f_oneway(*[group['LossRatio'] for group in risk_groups])
        
        # Calculate effect size
        grand_mean = self.data['LossRatio'].mean()
        total_ss = sum((self.data['LossRatio'] - grand_mean)**2)
        eta_squared = f_stat / (f_stat + len(self.data) - len(risk_groups))
        
        return p_value, p_value < 0.05, eta_squared
    
    def generate_visualizations(self, output_dir: str):
        """Generate visualizations for the analysis."""
        # Set style
        plt.style.use('bmh')  # Using a built-in matplotlib style
        
        # 1. Province Risk Distribution
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Province', y='LossRatio', data=self.data)
        plt.title('Loss Ratio Distribution by Province')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/province_risk_distribution.png')
        plt.close()
        
        # 2. Vehicle Type Risk Distribution
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='VehicleType', y='LossRatio', data=self.data)
        plt.title('Loss Ratio Distribution by Vehicle Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/vehicle_type_risk_distribution.png')
        plt.close()
        
        # 3. Risk Category Distribution
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='RiskCategory', y='LossRatio', data=self.data)
        plt.title('Loss Ratio Distribution by Risk Category')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/risk_category_distribution.png')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def generate_report(self, output_path: str):
        """Generate a comprehensive report of the analysis."""
        # Perform all tests
        province_p, province_reject, province_effect = self.test_province_risk_differences()
        vehicle_p, vehicle_reject, vehicle_effect = self.test_vehicle_type_risk_differences()
        risk_cat_p, risk_cat_reject, risk_cat_effect = self.test_risk_category_differences()
        
        # Calculate summary statistics
        province_stats = self.data.groupby('Province')['LossRatio'].agg(['mean', 'std', 'count'])
        vehicle_stats = self.data.groupby('VehicleType')['LossRatio'].agg(['mean', 'std', 'count'])
        risk_cat_stats = self.data.groupby('RiskCategory')['LossRatio'].agg(['mean', 'std', 'count'])
        
        # Generate report
        report = f"""# Risk Analysis Report

## Hypothesis Testing Results

### 1. Province Risk Differences
- Null Hypothesis: There are no risk differences across provinces
- p-value: {province_p:.4f}
- Decision: {'Reject' if province_reject else 'Fail to reject'} H0
- Effect Size (η²): {province_effect:.4f}

### 2. Vehicle Type Risk Differences
- Null Hypothesis: There are no risk differences between vehicle types
- p-value: {vehicle_p:.4f}
- Decision: {'Reject' if vehicle_reject else 'Fail to reject'} H0
- Effect Size (η²): {vehicle_effect:.4f}

### 3. Risk Category Differences
- Null Hypothesis: There are no significant differences between risk categories
- p-value: {risk_cat_p:.4f}
- Decision: {'Reject' if risk_cat_reject else 'Fail to reject'} H0
- Effect Size (η²): {risk_cat_effect:.4f}

## Summary Statistics

### Province-wise Loss Ratios
{province_stats.to_string()}

### Vehicle Type-wise Loss Ratios
{vehicle_stats.to_string()}

### Risk Category-wise Loss Ratios
{risk_cat_stats.to_string()}

## Business Recommendations

"""
        
        # Add business recommendations based on results
        if province_reject:
            report += """
### Province-based Recommendations
- Consider implementing region-specific premium adjustments
- Focus on high-risk provinces for risk mitigation strategies
- Develop targeted marketing campaigns for low-risk provinces
"""
        
        if vehicle_reject:
            report += """
### Vehicle Type-based Recommendations
- Implement vehicle type-specific premium adjustments
- Develop targeted risk management strategies for high-risk vehicle types
- Consider specialized insurance products for different vehicle categories
"""
        
        if risk_cat_reject:
            report += """
### Risk Category-based Recommendations
- Refine risk categorization methodology
- Develop specific risk mitigation strategies for each risk category
- Consider adjusting pricing strategies based on risk category performance
"""
        
        # Save report with UTF-8 encoding
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Report generated and saved to {output_path}")

if __name__ == "__main__":
    # Initialize analysis
    analysis = RiskAnalysis("Data/processed/insurance_summary_v1.csv")
    
    # Generate visualizations
    analysis.generate_visualizations("reports/figures")
    
    # Generate report
    analysis.generate_report("reports/hypothesis_testing_report.md") 