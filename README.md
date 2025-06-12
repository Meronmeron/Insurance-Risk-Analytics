# AlphaCare Insurance Solutions (ACIS) - Risk Analytics Project

## 🎯 Business Objective

AlphaCare Insurance Solutions (ACIS) is committed to developing cutting-edge risk and predictive analytics in the area of car insurance planning and marketing in South Africa. This project aims to analyze historical insurance claim data to:

- **Optimize marketing strategy** through data-driven insights
- **Discover "low-risk" targets** for premium reduction opportunities
- **Attract new clients** by offering competitive pricing to low-risk segments
- **Enhance risk assessment** capabilities for better underwriting decisions

## 📊 Dataset Overview

**Data Period**: February 2014 to August 2015  
**Domain**: Car Insurance Claims (South Africa)  
**Type**: Historical insurance claim and policy data

## 🗃️ Data Structure

### Insurance Policy Information

- `UnderwrittenCoverID` - Unique identifier for underwritten coverage
- `PolicyID` - Unique policy identifier

### Transaction Details

- `TransactionMonth` - Month of the transaction

### Client Demographics

- `IsVATRegistered` - VAT registration status
- `Citizenship` - Client citizenship
- `LegalType` - Legal entity type
- `Title` - Client title
- `Language` - Preferred language
- `Bank` - Banking institution
- `AccountType` - Type of bank account
- `MaritalStatus` - Marital status
- `Gender` - Gender

### Geographic Information

- `Country` - Country of residence
- `Province` - Province/state
- `PostalCode` - Postal code
- `MainCrestaZone` - Main geographical zone
- `SubCrestaZone` - Sub geographical zone

### Vehicle Specifications

- `ItemType` - Type of insured item
- `Mmcode` - Vehicle manufacturer code
- `VehicleType` - Type of vehicle
- `RegistrationYear` - Year of vehicle registration
- `Make` - Vehicle manufacturer
- `Model` - Vehicle model
- `Cylinders` - Number of cylinders
- `Cubiccapacity` - Engine cubic capacity
- `Kilowatts` - Engine power in kilowatts
- `Bodytype` - Vehicle body type
- `NumberOfDoors` - Number of doors
- `VehicleIntroDate` - Vehicle introduction date
- `CustomValueEstimate` - Custom value estimation
- `AlarmImmobiliser` - Alarm/immobilizer presence
- `TrackingDevice` - GPS tracking device presence
- `CapitalOutstanding` - Outstanding capital
- `NewVehicle` - New vehicle indicator
- `WrittenOff` - Write-off status
- `Rebuilt` - Rebuilt vehicle indicator
- `Converted` - Converted vehicle indicator
- `CrossBorder` - Cross-border usage indicator
- `NumberOfVehiclesInFleet` - Fleet size

### Insurance Plan Details

- `SumInsured` - Total insured amount
- `TermFrequency` - Premium payment frequency
- `CalculatedPremiumPerTerm` - Calculated premium per term
- `ExcessSelected` - Selected excess amount
- `CoverCategory` - Coverage category
- `CoverType` - Type of coverage
- `CoverGroup` - Coverage group
- `Section` - Policy section
- `Product` - Insurance product
- `StatutoryClass` - Statutory classification
- `StatutoryRiskType` - Statutory risk type

### Financial Information

- `TotalPremium` - Total premium amount
- `TotalClaims` - Total claims amount

## 🚀 Project Setup

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- Required Python packages (see requirements.txt)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd insurance-risk-analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
insurance-risk-analytics/
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned and processed data
│   └── external/               # External reference data
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_risk_modeling.ipynb
│   └── 04_marketing_insights.ipynb
├── src/
│   ├── data/                   # Data processing modules
│   ├── features/               # Feature engineering
│   ├── models/                 # ML models
│   └── visualization/          # Plotting utilities
├── reports/
│   ├── figures/                # Generated plots
│   └── final_report.md         # Final analysis report
├── requirements.txt
└── README.md
```

## 🔍 Analysis Approach

### Phase 1: Exploratory Data Analysis (EDA)

- Data quality assessment
- Missing value analysis
- Distribution analysis of key variables
- Correlation analysis
- Initial risk factor identification

### Phase 2: Data Preprocessing

- Data cleaning and validation
- Feature engineering
- Handling missing values
- Outlier detection and treatment
- Data transformation and scaling

### Phase 3: Risk Modeling

- Risk segmentation analysis
- Predictive modeling for claim probability
- Premium optimization models
- Low-risk customer identification

### Phase 4: Marketing Insights

- Customer segmentation
- Pricing strategy recommendations
- Market opportunity analysis
- Actionable business recommendations

## 📈 Key Performance Indicators (KPIs)

- **Claim Frequency**: Number of claims per policy
- **Claim Severity**: Average claim amount
- **Loss Ratio**: Claims to premium ratio
- **Customer Lifetime Value**: Long-term customer profitability
- **Risk Score**: Composite risk assessment metric

## 🎯 Expected Outcomes

1. **Risk Segmentation Model**: Classify customers into risk categories
2. **Premium Optimization Strategy**: Data-driven pricing recommendations
3. **Marketing Target List**: Identified low-risk prospects for acquisition
4. **Business Intelligence Dashboard**: Interactive visualization of key insights
5. **Actionable Recommendations**: Strategic recommendations for business growth

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/analysis-module`)
3. Commit your changes (`git commit -am 'Add new analysis module'`)
4. Push to the branch (`git push origin feature/analysis-module`)
5. Create a Pull Request

## 📝 License

This project is proprietary to AlphaCare Insurance Solutions (ACIS). All rights reserved.

## 📞 Contact

**Marketing Analytics Team**  
AlphaCare Insurance Solutions (ACIS)  
Email: analytics@alphacare.co.za  
Phone: +27 (0)11 XXX XXXX

---

**Project Status**: In Development  
**Last Updated**: December 2024  
**Next Milestone**: Exploratory Data Analysis Completion
