import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def clean_value(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    if val.startswith('(') and val.endswith(')'):
        val = '-' + val[1:-1]
    val = val.replace('$', '').replace(',', '').replace('%', '')
    try:
        return float(val)
    except ValueError:
        return np.nan

def remove_iqr_outliers(df, group_cols, value_col):
    def iqr_filter(group):
        q1 = group[value_col].quantile(0.25)
        q3 = group[value_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return group[(group[value_col] >= lower) & (group[value_col] <= upper)]

    grouped = df.groupby(group_cols)
    filtered = [iqr_filter(group) for _, group in grouped]
    return pd.concat(filtered, ignore_index=True)

def run_regression_pipeline(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[~df['Rules Based Pricing Description'].str.contains('KAROUN|ARZ', case=False, na=False)]

    df_long = df.melt(
        id_vars=['Rules Based Pricing Description', 'Rules Based Pricing Code', 'Data'],
        var_name='Week',
        value_name='Value'
    )
    df_long['Value'] = df_long['Value'].apply(clean_value)

    recent_weeks = sorted(df_long['Week'].unique())[-26:]

    recent_sales = df_long[
        (df_long['Data'] == 'Scanned Movement') &
        (df_long['Week'].isin(recent_weeks))
    ]

    sales_counts = (
        recent_sales
        .dropna(subset=['Value'])
        .groupby(['Rules Based Pricing Description', 'Rules Based Pricing Code'])['Week']
        .nunique()
    )

    valid_pairs = sales_counts[sales_counts == 26].index.tolist()

    long = df_long.set_index(['Rules Based Pricing Description', 'Rules Based Pricing Code']).loc[valid_pairs].reset_index()

    price_df = long[long['Data'] == 'Average Price']
    sales_df = long[long['Data'] == 'Scanned Movement']

    group_cols = ['Rules Based Pricing Description', 'Rules Based Pricing Code']

    price_df = remove_iqr_outliers(price_df, group_cols, 'Value')
    sales_df = remove_iqr_outliers(sales_df, group_cols, 'Value')

    price_pivot = price_df.pivot_table(index='Week', columns='Rules Based Pricing Description', values='Value', aggfunc='mean')
    sales_pivot = sales_df.pivot_table(index='Week', columns='Rules Based Pricing Description', values='Value', aggfunc='mean')

    price_pivot.columns = [f"Price_{col}" for col in price_pivot.columns]
    sales_pivot.columns = [f"Sales_{col}" for col in sales_pivot.columns]

    combined = pd.concat([price_pivot, sales_pivot], axis=1)
    combined.columns = [col.strip().replace('  ', ' ') for col in combined.columns]

    results = []
    price_cols = [col for col in combined.columns if col.startswith('Price_')]
    sales_cols = [col for col in combined.columns if col.startswith('Sales_')]

    for p_col in price_cols:
        p_item = p_col.replace('Price_', '').strip()
        for s_col in sales_cols:
            s_item = s_col.replace('Sales_', '').strip()

            if p_item == s_item:
                continue

            if p_col not in combined.columns or s_col not in combined.columns:
                continue

            data_pair = combined[[p_col, s_col]].dropna()
            if data_pair[p_col].nunique() <= 1 or data_pair[s_col].nunique() <= 1:
                continue

            X = sm.add_constant(data_pair[p_col])
            y = data_pair[s_col]

            model = sm.OLS(y, X).fit()
            coef = model.params[p_col]
            pval = model.pvalues[p_col]
            r2 = model.rsquared

            if pval < 1:  # Allow all relationships
                results.append({
                    'price_item': p_item,
                    'sales_item': s_item,
                    'coefficient_per_dollar': coef,
                    'coefficient_per_dime': coef * 0.10,
                    'p_value': pval,
                    'r_squared': r2,
                    'n_points': len(data_pair)
                })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df['abs_coef'] = results_df['coefficient_per_dollar'].abs()
        results_df = results_df.sort_values(by='r_squared', ascending=False)

    return results_df
