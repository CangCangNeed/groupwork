import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import linregress, ttest_ind, mannwhitneyu, chi2_contingency, shapiro
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
try:
    from transformers import pipeline  
except ImportError:
    pipeline = None


def clean_gender_column(df, column_name='Gender'):
    """
    Clean the Gender column in the DataFrame by standardizing values and handling missing values.
    """
    gender_map = {
        'F': 'Female',
        'Female': 'Female',
        'M': 'Male',
        'Male': 'Male',  
        'Other': 'Other',
        'Prefer not to say': 'Other',
        'nan': np.nan,
        '': np.nan  
    }

    df[column_name] = df[column_name].replace(gender_map)

    mode_gender = df[column_name].mode()[0] if not df[column_name].mode().empty else 'Unknown'
    df[column_name] = df[column_name].fillna(mode_gender)

    #print(f"Cleaned '{column_name}' column. Most common gender: {mode_gender}")
    return df

import numpy as np

def clean_IncomeLevel_column(df, column_name='IncomeLevel'):
    """
    Clean the IncomeLevel column in the DataFrame by standardizing values and handling missing values.
    """
    IncomeLevel_map = {
        'H': 'High',
        'High': 'High',
        'L': 'Low',
        'Low': 'Low',
        'Medium': 'Medium',
        'Very High': 'Very High',
        'nan': np.nan,  
        '': np.nan      
    }

    # Replace the values in the column based on the mapping
    df[column_name] = df[column_name].replace(IncomeLevel_map)

    # Calculate the mode (most common value) in the column
    if df[column_name].notna().sum() > 0:  # Ensure there are valid values to calculate the mode
        mode_IncomeLevel = df[column_name].mode()[0]
    else:
        mode_IncomeLevel = 'Unknown'  

    # Fill missing values (NaN) with the mode
    df[column_name] = df[column_name].fillna(mode_IncomeLevel)

    #print(f"Cleaned '{column_name}' column. Most common IncomeLevel: {mode_IncomeLevel}")
    return df

class DataInspection:
    def __init__(self):
        self.df = None

    def load_csv(self, file_path):
        """
        Load the CSV file and handle missing values and column cleaning.
        """
        try:
            self.df = pd.read_csv(file_path)
            print("Dataset loaded successfully.")
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
        
        # Handle missing values and clean data
        for col in self.df.columns:
            self.handle_missing_values(col)
        return True

    def handle_missing_values(self, col):
        """
        Handle missing values and outliers for any dataset column, depending on its type.
        """
        # Check if the column exists in the DataFrame
        if col not in self.df.columns:
            print(f"Column '{col}' does not exist in the DataFrame.")
            return False  

        # Convert any 'nan' strings to actual NaN values
        self.df[col] = self.df[col].replace('nan', np.nan)
        
        # Calculate missing percentage
        missing_percentage = self.df[col].isnull().mean() * 100
        #print(f"\nMissing percentage for column '{col}': {missing_percentage:.2f}%")

        # Unique identifier: Drop rows with missing CustomerID
        if 'ID' in col and missing_percentage > 0:
            self.df = self.df.dropna(subset=[col])
            #print(f"Dropped rows with missing values in column '{col}'.")
            return True  # Rows were dropped or no missing values

        # Handle Gender column
        if col == 'Gender':
            self.df = clean_gender_column(self.df, col)
            #print(f"Handled missing values in 'Gender' column.")
            return True
        # Handle IncomeLevel column
        if col == 'IncomeLevel':
            self.df = clean_IncomeLevel_column(self.df, col)
            #print(f"Handled missing values in 'IncomeLevel' column.")
            return True

        # Handle date columns
        if pd.api.types.is_datetime64_any_dtype(self.df[col]):
            if missing_percentage > 20:  # Arbitrary threshold
                self.df = self.df.drop(columns=[col])
                #print(f"Column '{col}' dropped due to high missing percentage.")
                return True

            most_common_date = self.df[col].mode()[0] if not self.df[col].mode().empty else pd.NaT
            self.df[col] = self.df[col].fillna(most_common_date)
            #print(f"Filled missing values in '{col}' with mode: {most_common_date}")
            return True

        # Handle missing values for numeric columns
        elif pd.api.types.is_numeric_dtype(self.df[col]):
            # Force conversion of text-based numbers to numeric (invalid strings become NaN)
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

            if missing_percentage > 50:
                #print(f"Column '{col}' dropped due to more than 50% missing values.")
                self.df = self.df.drop(columns=[col])
                return False  # Column was dropped

            # Fill missing values with the median for numeric columns
            median_value = self.df[col].median()
            self.df[col] = self.df[col].fillna(median_value)
            #print(f"Filled missing values in numeric column '{col}' with median: {median_value}")
            return True

        # Handle missing values for categorical or text columns
        else:
            # Convert all entries to string to handle 'nan' and other non-standard missing values
            self.df[col] = self.df[col].astype(str).replace('nan', np.nan)

            if missing_percentage > 50:
                #print(f"Column '{col}' dropped due to more than 50% missing values.")
                self.df = self.df.drop(columns=[col])
                return False  # Column was dropped

            # Fill missing values with the mode for categorical/text columns
            mode_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
            self.df[col] = self.df[col].fillna(mode_value)
            #print(f"Filled missing values in categorical/text column '{col}' with mode: {mode_value}")
            return True
        
    def show_variable_statistics(self):
        """
        Show Variable, Type, Mean/Median/Mode, Kurtosis, and Skewness for each column.
        """
        print(f"\n{'Variable':<30}{'Type':<20}{'Mean/Median/Mode':<30}{'Kurtosis':<25}{'Skewness':<25}")
        print("="*130)
        
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)  # Data type of the column
            if pd.api.types.is_numeric_dtype(dtype):
                col_type = 'Ratio'
                central_tendency = self.df[col].mean()  # Use mean for numeric columns
            else:
                col_type = 'Ordinal' if self.df[col].nunique() < 10 else 'Nominal'
                central_tendency = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'  # Mode for categorical
            
            # Calculate Kurtosis and Skewness for numeric columns
            kurtosis = self.df[col].kurt() if pd.api.types.is_numeric_dtype(self.df[col]) else 'N/A'
            skewness = self.df[col].skew() if pd.api.types.is_numeric_dtype(self.df[col]) else 'N/A'

            print(f"{col:<30}{col_type:<20}{central_tendency:<30}{kurtosis:<25}{skewness:<25}")
        
    def show_analysis_menu(self):
        """
        Show analysis options for the user and allow them to select the analysis they want to perform.
        """
        while True:
            print("\nHow do you want to analyze your data?")
            print("1. Plot variable distribution")
            print("2. Conduct ANOVA")
            print("3. Conduct t-Test")
            print("4. Conduct chi-Square")
            print("5. Conduct Regression")
            print("6. Conduct Sentiment Analysis")
            print("7. Quit")
            
            choice = input("Enter your choice (1 - 7): ").strip()
            if choice == "1":
                self.plot_variable_distribution()
            elif choice == "2":
                self.perform_anova()
            elif choice == "3":
                self.perform_t_test()
            elif choice == "4":
                self.perform_chi_square()
            elif choice == "5":
                self.perform_regression()
            elif choice == "6":
                self.perform_sentiment_analysis()
            elif choice == "7":
                print("Exiting program.")
                break
            else:
                print("Invalid choice. Please choose a number between 1 and 7.")

    def plot_variable_distribution(self):
        """
        Plot the distribution of a selected numeric variable. This includes:
        1. A standalone histogram with KDE.
        2. A combined plot with both a histogram and Q-Q plot.
        3. Analyze the normality characteristics based on skewness and kurtosis.
        """
        # Get numeric columns in the dataset
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        # Display available numeric columns
        print("\nAvailable numeric columns for plotting distribution:")
        for idx, col in enumerate(numeric_columns, 1):
            print(f"{idx}. {col}")

        # Ask the user to select a column
        column_index = int(input("Select a column to plot its distribution (by index): ")) - 1
        column_name = numeric_columns[column_index]

        # Get the data from the selected column
        data = self.df[column_name].dropna()

        # First plot: standalone histogram with KDE
        plt.figure(figsize=(8, 5))
        sns.histplot(data, kde=True)
        plt.title(f'Distribution of {column_name}')
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        plt.show()

        fig, ax = plt.subplots(figsize=(8, 6))

        # Q-Q plot for normality check
        stats.probplot(data, dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot of {column_name}')
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Sample Quantiles')

        # Display the Q-Q plot
        plt.tight_layout()
        plt.show()

        # Calculate skewness and kurtosis
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)

        # Analyze the normality characteristics
        normality_type = self.analyze_normality(skewness, kurtosis)

        # Print the result for normality
        print(f"\nNormality Analysis for '{column_name}':")
        print(f"Skewness: {skewness:.4f}, Kurtosis: {kurtosis:.4f}")
        print(f"Data Normality is: {normality_type}")

    def analyze_normality(self, skewness, kurtosis):
        """
        Analyze the normality of the data based on skewness and kurtosis values.
        Return a description of the normality type.
        """
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            return "Normal"
        elif skewness > 0.5:
            return "Right-skewed"
        elif skewness < -0.5:
            return "Left-skewed"
        elif kurtosis > 1:
            return "Leptokurtic (Peaked distribution)"
        elif kurtosis < -1:
            return "Platykurtic (Flat distribution)"
        elif len(self.df) > 5000 and (kurtosis > 0.5 and kurtosis < 1 and skewness == 0):
            return "Bimodal"
        else:
            return "Normal or Slight Deviation"


    def perform_anova(self):
        """
        Perform ANOVA or Kruskal-Wallis test depending on the normality of the selected continuous variable.
        """
        # Display available variables for ANOVA
        print("\nFor ANOVA, following are the variables available:")
        print(f"\n{'Variable':<30}{'Type':<20}")
        print("="*50)

        for col in self.df.columns:
            dtype = self.df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                col_type = 'Ratio'
            else:
                col_type = 'Ordinal' if self.df[col].nunique() < 10 else 'Nominal'
            print(f"{col:<30}{col_type:<20}")

        # Prompt user to enter a continuous and categorical variable
        continuous_var = input("\nEnter a continuous (interval/ratio) variable: ").strip()
        categorical_var = input("Enter a categorical (ordinal/nominal) variable: ").strip()

        # Check if the selected continuous variable is normally distributed
        data = self.df[continuous_var].dropna()
        self.plot_qq_histogram_combined(data, continuous_var)
    
        stat, p_value = self.check_normality(data)

        # If Anderson-Darling test is used (p_value is an array)
        if isinstance(p_value, (list, np.ndarray)):
            print(f"Anderson-Darling Test: Stat={stat}, Critical Values={p_value}")
            # Compare the statistic with the critical value at 5% significance level
            if stat > p_value[2]:  # Compare with 5% significance level critical value
                print(f"'{continuous_var}' is not normally distributed, as shown in the Q-Q plot…")
                print("Performing Kruskal-Wallis Test instead…")
                # Perform Kruskal-Wallis Test
                stat, p_value = self.perform_anova_or_kruskal(continuous_var, categorical_var, skewed=True)
                test_name = "Kruskal-Wallis"
            else:
                print(f"'{continuous_var}' is normally distributed.")
                # Perform ANOVA
                stat, p_value = self.perform_anova_or_kruskal(continuous_var, categorical_var, skewed=False)
                test_name = "ANOVA"
        else:
            # If Shapiro-Wilk test is used (p_value is a single value)
            if p_value < 0.05:
                print(f"'{continuous_var}' is not normally distributed, as shown in the Q-Q plot…")
                print("Performing Kruskal-Wallis Test instead…")
                # Perform Kruskal-Wallis Test
                stat, p_value = self.perform_anova_or_kruskal(continuous_var, categorical_var, skewed=True)
                test_name = "Kruskal-Wallis"
            else:
                print(f"'{continuous_var}' is normally distributed.")
                # Perform ANOVA
                stat, p_value = self.perform_anova_or_kruskal(continuous_var, categorical_var, skewed=False)
                test_name = "ANOVA"

        # Display results
        print(f"\n{test_name} Result:")
        print(f"{test_name} Statistic: {stat}")
        print(f"p-value: {p_value}")

        # Check significance
        if p_value < 0.05:
            print("Result is statistically significant.")
            print("Therefore, your Null Hypothesis is rejected.")
            print(f"There is a statistically significant difference in the average '{continuous_var}' across the categories of '{categorical_var}'")
        else:
            print("Result is not statistically significant.")
            print("Therefore, we fail to reject the Null Hypothesis.")


    
    def plot_qq_histogram_combined(self, data, title):
        """
        Plot Q-Q plot and histogram for normality check.
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Q-Q plot
        stats.probplot(data, dist="norm", plot=ax[0])
        ax[0].set_title(f'Q-Q Plot of {title}')
        ax[0].set_xlabel('Theoretical Quantiles')
        ax[0].set_ylabel('Sample Quantiles')
        ax[0].grid(True)

        # Histogram
        sns.histplot(data, kde=True, ax=ax[1], bins='auto', stat='density')
        ax[1].set_title(f'Histogram of {title}')
        ax[1].set_xlabel(title)
        ax[1].set_ylabel('Density')

        fig.suptitle(f'Normality Check for {title}', fontsize=16)
        plt.tight_layout()
        plt.show()

    def perform_anova_or_kruskal(self, continuous_var, categorical_var, skewed):
        """
        Perform ANOVA or Kruskal-Wallis test based on the normality of the continuous variable.
        """
        continuous_data = self.df[continuous_var].dropna()
        categorical_data = self.df[categorical_var].dropna()

        # Group data by categorical variable
        grouped_data = [continuous_data[self.df[categorical_var] == group] for group in np.unique(categorical_data)]

        if skewed:
            stat, p_value = stats.kruskal(*grouped_data)
        else:
            stat, p_value = stats.f_oneway(*grouped_data)

        return stat, p_value

    def check_normality(self, data):
        """
        Check normality using Anderson-Darling test for large sample sizes.
        """
        if len(data) > 5000:
            result = stats.anderson(data)
            print(f"Anderson-Darling Test: Stat={result.statistic}, Critical Values={result.critical_values}")
            return result.statistic, result.critical_values
        else:
            stat, p_value = stats.shapiro(data)
            print(f"Shapiro-Wilk Test: Stat={stat}, p-value={p_value}")
            return stat, p_value
    
    def perform_t_test(self):
        """
            Perform t-Test or Mann-Whitney U Test (if non-parametric).
        """
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

        print("\nAvailable numeric columns:")
        for idx, col in enumerate(numeric_columns, 1):
            print(f"{idx}. {col}")
        numeric_column_index = int(input("Select a numeric column (by index): ")) - 1
        numeric_column = numeric_columns[numeric_column_index]

        print("\nAvailable categorical columns:")
        for idx, col in enumerate(categorical_columns, 1):
            print(f"{idx}. {col}")
        categorical_column_index = int(input("Select a binary categorical column (by index): ")) - 1
        categorical_column = categorical_columns[categorical_column_index]

        # Ensure binary categorical variable (two groups)
        unique_values = self.df[categorical_column].unique()
        if len(unique_values) != 2:
            print("The selected categorical column does not have exactly two groups.")
            return

        group1 = self.df[self.df[categorical_column] == unique_values[0]][numeric_column].dropna()
        group2 = self.df[self.df[categorical_column] == unique_values[1]][numeric_column].dropna()

        # Perform normality test
        _, p_value1 = shapiro(group1)
        _, p_value2 = shapiro(group2)

        if p_value1 > 0.05 and p_value2 > 0.05:
            # Use t-Test
            t_stat, p_val = ttest_ind(group1, group2)
            print(f"t-Test result: t-statistic = {t_stat}, p-value = {p_val}")
        else:
            # Use Mann-Whitney U Test (non-parametric)
            u_stat, p_val = mannwhitneyu(group1, group2)
            print(f"Mann-Whitney U Test result: U-statistic = {u_stat}, p-value = {p_val}")
        
        # Interpretation
        if p_val < 0.05:
            print("Significant difference found between the two groups.")
        else:
            print("No significant difference found between the two groups.")
    
    def perform_chi_square(self):
        """
        Perform chi-Square analysis for categorical variables.
        """
        categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

        print("\nAvailable categorical columns:")
        for idx, col in enumerate(categorical_columns, 1):
            print(f"{idx}. {col}")

        col1_index = int(input("Select the first categorical column (by index): ")) - 1
        col2_index = int(input("Select the second categorical column (by index): ")) - 1

        col1 = categorical_columns[col1_index]
        col2 = categorical_columns[col2_index]

        contingency_table = pd.crosstab(self.df[col1], self.df[col2])
        chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)

        print(f"Chi-Square result: chi2-statistic = {chi2_stat}, p-value = {p_val}")

        if p_val < 0.05:
            print("There is a significant association between the variables.")
        else:
            print("No significant association between the variables.")
    
    def perform_regression(self):
        """
        Perform linear regression analysis.
        """
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        print("\nAvailable numeric columns:")
        for idx, col in enumerate(numeric_columns, 1):
            print(f"{idx}. {col}")
        dependent_idx = int(input("Select the dependent variable (Y) by index: ")) - 1
        independent_idx = int(input("Select the independent variable (X) by index: ")) - 1

        y = self.df[numeric_columns[dependent_idx]].dropna()
        x = self.df[numeric_columns[independent_idx]].dropna()

        # Ensure X and Y have the same length after dropping NaN values
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]

        # Add constant term for intercept
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        print(model.summary())
        
    def perform_sentiment_analysis(self):
        """
        Conduct Sentiment Analysis on suitable text columns.
        """
        print("Looking for text data in your dataset…")

        # Identify text columns
        text_columns = self.df.select_dtypes(include='object')

        if text_columns.empty:
            print("Sorry, your dataset does not have any suitable text data.")
            print("Therefore, Sentiment Analysis is not possible.")
            print("Returning to previous menu…")
            return

        # Check if there are columns with sufficiently long text data
        suitable_text_columns = []
        for col in text_columns.columns:
            avg_length = text_columns[col].apply(lambda x: len(str(x)) if pd.notnull(x) else 0).mean()
            if avg_length > 30:  # Arbitrary threshold for "sufficiently long" text
                suitable_text_columns.append(col)

        if not suitable_text_columns:
            print("Sorry, your dataset does not have suitable length text data.")
            print("Therefore, Sentiment Analysis is not possible.")
            print("Returning to previous menu…")
            return

        # Display available text columns
        print("\nAvailable text columns for Sentiment Analysis:")
        for idx, col in enumerate(suitable_text_columns, 1):
            print(f"{idx}. {col}")

        # Let the user select a text column for analysis
        column_index = int(input("Select a text column to analyze (by index): ")) - 1
        column_name = suitable_text_columns[column_index]

        # Display text data to the user for review
        print(f"\nText data in the selected column '{column_name}':")
        print(self.df[column_name].head())  # Show the first few rows

        # Ask the user to choose a sentiment analysis method
        print("\nChoose the type of sentiment analysis:")
        print("1. VADER")
        print("2. TextBlob")
        choice = input("Enter your choice (1/2): ").strip()

        # Perform sentiment analysis based on the user's choice
        if choice == '1':
            self.vader_sentiment_analysis(self.df[column_name])
        elif choice == '2':
            self.textblob_sentiment_analysis(self.df[column_name])
        else:
            print("Invalid choice. Returning to previous menu.")

    def vader_sentiment_analysis(self, data):
        """
        Perform VADER sentiment analysis on the selected text column.
        """
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        sentiments = []

        for text in data:
            vs = analyzer.polarity_scores(str(text))
            scores.append(vs['compound'])
            if vs['compound'] >= 0.05:
                sentiments.append('positive')
            elif vs['compound'] <= -0.05:
                sentiments.append('negative')
            else:
                sentiments.append('neutral')

        result_df = pd.DataFrame({'Text': data, 'Score': scores, 'Sentiment': sentiments})
        print("\nSentiment Analysis Results (VADER):")
        print(result_df)

    def textblob_sentiment_analysis(self, data):
        """
        Perform TextBlob sentiment analysis on the selected text column.
        """
        scores = []
        sentiments = []
        subjectivities = []

        for text in data:
            blob = TextBlob(str(text))
            score = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            scores.append(score)
            subjectivities.append(subjectivity)

            if score > 0:
                sentiments.append('positive')
            elif score < 0:
                sentiments.append('negative')
            else:
                sentiments.append('neutral')

        result_df = pd.DataFrame({'Text': data, 'Score': scores, 'Sentiment': sentiments, 'Subjectivity': subjectivities})
        print("\nSentiment Analysis Results (TextBlob):")
        print(result_df)
    
if __name__ == "__main__":
    inspector = DataInspection()
    file_path = input("Enter the path of the dataset (CSV file): ").strip()
    
    if inspector.load_csv(file_path):
        inspector.show_variable_statistics()

    inspector.show_analysis_menu()