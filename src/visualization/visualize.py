import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def fraud_vs_nonfraud_plot(df, x_col, y_col):
    """
    Function to create a countplot of fraud vs non-fraud cases.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    x_col (str): The column to plot on the x-axis.
    y_col (str): The column to plot on the y-axis.

    Sample usage:
    fraud_vs_nonfraud_plot(df, 'insured_education_level', 'fraud_reported')

    """
    plt.figure(figsize=(10, 6))
    
    # Count of records for each x/y combination
    sns.countplot(data=df, x=x_col, hue=y_col,order=sorted(df[x_col].unique()))

    plt.title(f'Fraud vs Non-Fraud Claims by {x_col}')
    plt.xlabel(x_col)
    plt.ylabel('Number of Claims')
    plt.legend(title=y_col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_fraud_cases(df, category_col, original_col, category_order, bins):
    """
    Function to plot fraud cases by a specified categorical column.
    
    Parameters:
    fraud_counts (pd.DataFrame): DataFrame containing fraud data with the specified categorical column and counts.
    category_col (str): The name of the categorical column to group by (e.g., 'age_range', 'months_as_customer_range').
    category_order (list): Ordered list of categories for sorting.

    Sample usage:
    plot_fraud_cases(df, 'age_range','age', labels, list(range(15,66,5)))

    """
    df[category_col] = pd.cut(df[original_col], bins=bins, labels=category_order, right=False)
    df[category_col] = df[category_col].astype(str)

    fraud_counts = df.groupby(category_col)['fraud_reported'].value_counts().unstack().fillna(0)
    fraud_counts.columns = ['Non-Fraud', 'Fraud']
    fraud_counts = fraud_counts.reset_index()
    # Ensure the specified column is categorical and ordered
    fraud_counts[category_col] = pd.Categorical(fraud_counts[category_col], categories=category_order, ordered=True)
    fraud_counts = fraud_counts.sort_values(category_col)

    # Plot
    plt.figure(figsize=(8, 4))
    fraud_counts.set_index(category_col).plot(kind='bar', stacked=False, figsize=(12, 6), colormap='coolwarm')

    plt.xlabel(category_col, fontsize=12)
    plt.ylabel('Number of Fraud Cases', fontsize=12)
    plt.title(f'Fraud Cases Reported by {category_col}', fontsize=14)
    plt.legend(['Non-fraud', 'Fraud'])
    plt.xticks(rotation=45)
    plt.show()


def plot_statewise_fraud_cases(df, fraud_column, state_column, title="State-wise Fraud Cases Count"):
    """
    Function to plot state-wise fraud cases on a choropleth map.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    fraud_column (str): The column indicating whether fraud was reported (e.g., 'fraud_reported').
    state_column (str): The column containing state information (e.g., 'incident_state').
    title (str): The title of the plot (default is "State-wise Fraud Cases Count").

    Sample usage:
    plot_statewise_fraud_cases(df, 'fraud_reported', 'incident_state', 'Fraud Cases by State')
    """
    # Filter for fraud cases
    fraud_cases = df[df[fraud_column] == 'Y']
    
    # Count fraud cases by state
    state_counts = fraud_cases[state_column].value_counts().reset_index()
    state_counts.columns = ["state", "count"]
    
    # Create the choropleth map
    fig = px.choropleth(
        state_counts,
        locations='state',
        locationmode="USA-states",
        color='count',
        scope="usa",
        color_continuous_scale="Reds",
        title=title
    )
    
    # Update layout to set figure size
    fig.update_layout(
        width=800,
        height=600,
    )
    
    # Show the plot
    fig.show()


def plot_relationship(df, x_col, y_col, hue_col):
    """
    Function to visualize the relationship between two columns, grouped by a third column (hue).
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    x_col (str): The column to plot on the x-axis.
    y_col (str): The column to plot on the y-axis.
    hue_col (str): The column to group by (used as hue in the plot).

    Sample usage:
    plot_relationship(df, 'property_damage', 'property_claim', 'fraud_reported')
    """

    # Set plot size
    plt.rcParams['figure.figsize'] = (8, 5)

    # Create the strip plot
    sns.stripplot(
        x=df[x_col],
        y=df[y_col],
        hue=df[hue_col],
        palette='bone'
    )
    
    # Add title and show the plot
    plt.title(f'{x_col} vs {y_col}')
    plt.xticks(rotation=45)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(title=hue_col)
    plt.show()

