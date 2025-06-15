import joblib
import pandas as pd
import sys
import types
import warnings

# Define bool_to_int function explicitly
def bool_to_int(x):
    return x.astype(int)

# Create the full module structure dynamically
def create_module_structure():
    """Create the entire module structure from src.training_pipeline to satisfy unpickling"""
    # Create base 'src' module
    if 'src' not in sys.modules:
        src_module = types.ModuleType('src')
        sys.modules['src'] = src_module
    
    # Create 'training_pipeline' under src
    if 'src.training_pipeline' not in sys.modules:
        training_pipeline = types.ModuleType('src.training_pipeline')
        sys.modules['src.training_pipeline'] = training_pipeline
    
    # Create 'proc_data' under src.training_pipeline
    if 'src.training_pipeline.proc_data' not in sys.modules:
        proc_data = types.ModuleType('src.training_pipeline.proc_data')
        sys.modules['src.training_pipeline.proc_data'] = proc_data
    
    # Add the bool_to_int function to the module
    sys.modules['src.training_pipeline.proc_data'].bool_to_int = bool_to_int

class ChurnSegmenter:
    def __init__(self, artifact_path="../artifacts/segmentation/segmentation_artifacts.pkl"):
        """
        Initialize the segmenter by loading trained artifacts
        
        Parameters:
        artifact_path : path to saved segmentation artifacts
        """
        # Create module structure before loading artifacts
        create_module_structure()
        
        # Load artifacts
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore version mismatch warnings
            artifacts = joblib.load(artifact_path)
        self.model = artifacts['kmeans']
        self.scaler = artifacts['scaler']
        self.features = artifacts['features']
        self.profiles = artifacts['profiles']
        self.continuous = ['tenure', 'MonthlyCharges']
        
    def preprocess_customer(self, customer_data):
        """
        Prepare customer data for segmentation
        """
        # Create a copy of the customer data
        processed = customer_data[self.features].copy()
        
        # Scale continuous features
        if self.continuous:
            processed[self.continuous] = self.scaler.transform(
                processed[self.continuous]
            )
        
        return processed
    
    def assign_segment(self, customer_data):
        """
        Assign a churning customer to a segment
        """
        # Preprocess customer data
        processed = self.preprocess_customer(customer_data)
        
        # Predict segment
        segment = self.model.predict(processed)[0]
        
        return segment
    
    def generate_report(self, segment):
        """
        Generate business report for the assigned segment
        """
        # Get segment profile
        profile = self.profiles[self.profiles['Segment'] == segment].iloc[0]
        
        # Generate report text
        report = f"SEGMENT REPORT: {profile['Segment_Label']}\n"
        report += "="*50 + "\n\n"
        report += f"Segment Size: {profile['Size']} customers ({profile['Pct_Total']:.1f}% of churned)\n\n"
        
        # Key characteristics
        report += "KEY CHARACTERISTICS:\n"
        report += f"- Average Tenure: {profile['tenure_mean']:.1f} months\n"
        report += f"- Average Monthly Charges: ${profile['MonthlyCharges_mean']:.2f}\n"
        report += f"- Senior Citizens: {profile['SeniorCitizen_prop']*100:.1f}%\n"
        report += f"- With Dependents: {profile['Dependents_prop']*100:.1f}%\n"
        report += f"- Fiber Optic Users: {profile['InternetService_Fiber optic_prop']*100:.1f}%\n"
        report += f"- Month-to-Month Contracts: {profile['Contract_Month-to-month_prop']*100:.1f}%\n\n"
        
        # Service usage
        report += "SERVICE USAGE:\n"
        report += f"- Multiple Lines: {profile['MultipleLines_Yes_prop']*100:.1f}%\n"
        report += f"- Online Backup: {profile['OnlineBackup_Yes_prop']*100:.1f}%\n"
        report += f"- Online Security: {profile['OnlineSecurity_Yes_prop']*100:.1f}%\n"
        report += f"- Device Protection: {profile['DeviceProtection_Yes_prop']*100:.1f}%\n"
        report += f"- Tech Support: {profile['TechSupport_Yes_prop']*100:.1f}%\n"
        report += f"- Paperless Billing: {profile['PaperlessBilling_prop']*100:.1f}%\n\n"
        
        # Interpretation
        label_parts = profile['Segment_Label'].split('-')
        report += "SEGMENT INTERPRETATION:\n"
        report += f"This segment represents {label_parts[0].lower()}-tenure customers with {label_parts[1].lower()} spending patterns "
        report += f"and {label_parts[2].lower()} service adoption.\n\n"
        
        # Risk factors
        report += "CHURN RISK FACTORS:\n"
        if profile['Contract_Month-to-month_prop'] > 0.7:
            report += "- High proportion of month-to-month contracts (low commitment)\n"
        if profile['tenure_mean'] < 12:
            report += "- New customers with low tenure (highest churn risk period)\n"
        if profile['MonthlyCharges_mean'] > 80 and profile['Contract_Month-to-month_prop'] > 0.6:
            report += "- High monthly charges without long-term commitment\n"
        if profile['InternetService_Fiber optic_prop'] > 0.8 and profile['OnlineSecurity_Yes_prop'] < 0.3:
            report += "- Fiber optic users without adequate security services\n"
        report += "\n"
        
        # Recommendations
        report += "RECOMMENDED RETENTION STRATEGIES:\n"
        report += f"1. TARGETED OFFERS: {self.get_personalized_offer(profile)}\n"
        report += f"2. SERVICE IMPROVEMENTS: {self.get_service_recommendation(profile)}\n"
        report += f"3. CONTRACT INCENTIVES: {self.get_contract_recommendation(profile)}\n"
        report += f"4. CUSTOMER ENGAGEMENT: {self.get_engagement_strategy(profile)}\n"
        report += f"5. LOYALTY PROGRAM: {self.get_loyalty_recommendation(profile)}\n\n"
        
        # Business impact
        annual_revenue = profile['Size'] * profile['MonthlyCharges_mean'] * 12
        report += "BUSINESS IMPACT:\n"
        report += f"Preventing churn in this segment could retain ${annual_revenue:,.0f} in annual revenue.\n"
        
        return report
    
    # Helper functions for recommendations
    def get_personalized_offer(self, profile):
        if profile['MonthlyCharges_mean'] > 80:
            return "Premium bundle discount: Offer 20% off premium services for 6 months"
        elif profile['InternetService_Fiber optic_prop'] > 0.7:
            if profile['OnlineSecurity_Yes_prop'] < 0.3:
                return "Free security bundle: Include online security for 3 months"
            return "Speed upgrade: Free internet speed boost for loyal customers"
        elif profile['Contract_Month-to-month_prop'] > 0.8:
            return "Commitment incentive: $10/month discount for 1-year contract"
        return "Value package: Custom bundle of most-used services at 15% discount"
    
    def get_service_recommendation(self, profile):
        low_services = []
        if profile['OnlineSecurity_Yes_prop'] < 0.3:
            low_services.append("security")
        if profile['DeviceProtection_Yes_prop'] < 0.3:
            low_services.append("device protection")
        if profile['TechSupport_Yes_prop'] < 0.3:
            low_services.append("tech support")
        
        if low_services:
            return f"Educate customers on benefits of {'/'.join(low_services)} through personalized tutorials"
        return "Enhance existing services with premium features for free trial period"
    
    def get_contract_recommendation(self, profile):
        if profile['Contract_Month-to-month_prop'] > 0.7:
            return "Offer tiered discounts: 10% for 1-year, 15% for 2-year contracts"
        return "Introduce flexible long-term contracts with price-lock guarantee"
    
    def get_engagement_strategy(self, profile):
        if profile['SeniorCitizen_prop'] > 0.25:
            return "Senior-focused support: Dedicated helpline and in-person tech assistance"
        elif profile['Dependents_prop'] > 0.3:
            return "Family plans: Create bundled services for households with multiple users"
        return "Proactive check-ins: Monthly usage reports and optimization suggestions"
    
    def get_loyalty_recommendation(self, profile):
        if profile['tenure_mean'] > 24:
            return "Loyalty rewards: Exclusive benefits for long-term customers (early upgrades, priority support)"
        return "Points system: Earn redeemable points for each month of service"

# Full inference pipeline
def churn_management_pipeline(raw_customer_data):
    """
    Complete churn management pipeline:
    1. Clean raw customer data using the saved pipeline
    2. Predict churn probability
    3. If churning, assign to segment and generate report
    
    Parameters:
    raw_customer_data : DataFrame with single customer record (raw format)
    churn_prediction_model : Trained churn prediction model
    segmenter : Initialized ChurnSegmenter instance
    cleaning_pipeline_path : Path to saved cleaning pipeline
    
    Returns:
    dict: churn probability, segment assignment, report, and cleaned data
    """
    # Create the full module structure before loading
    create_module_structure()
        # Initialize components 
    cleaning_pipeline_path="../data/processed/pipeline-parametric_models.pkl"
    segmenter = ChurnSegmenter()

    # Load churn prediction model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        churn_prediction_model = joblib.load("../artifacts/churn_prediction/logistic_regression_model.pkl")

    # Load cleaning pipeline
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore version mismatch warnings
        cleaning_pipeline = joblib.load(cleaning_pipeline_path)
    cleaned_data = cleaning_pipeline.transform(raw_customer_data.copy())
    # Predict churn probability
    churn_prob = churn_prediction_model.predict_proba(cleaned_data)[0][1]
    
    result = {
        'churn_probability': churn_prob,
        'is_churning': False,
        'segment': None,
        'report': None,
        'cleaned_data': cleaned_data
    }
    
    # Check if customer is churning (using 0.5 threshold)
    if churn_prob >= 0.5:
        result['is_churning'] = True
        
        # Assign to segment
        segment = segmenter.assign_segment(cleaned_data)
        result['segment'] = segment
        
        # Generate report
        report = segmenter.generate_report(segment)
        result['report'] = report
    
    return result
if __name__ == "__main__":

    # Example raw customer data (matching original 21 columns)
    # Run through pipeline
    result = churn_management_pipeline(
        raw_customer_data = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": False,
    "Partner": True,
    "Dependents": False,
    "tenure": 1,
    "PhoneService": False,
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": True,
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}])

    )

    # Output results    
    print(f"Churn Probability: {result['churn_probability']:.2f}")
    print(f"Cleaned Data Shape: {result['cleaned_data'].shape}")
    if result['is_churning']:
        print(f"Assigned to Segment: {result['segment']}")
        print("\nRetention Strategy Report:")
        print(result['report'])
    

