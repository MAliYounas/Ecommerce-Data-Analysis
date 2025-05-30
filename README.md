
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error  # removed unused r2_score import
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from datetime import datetime
import scipy.stats as stats
from streamlit_option_menu import option_menu

# ================================================
# PATHS
# ================================================
customers_path = "C:/Users/hp/Downloads/archive (1)/customer_details.csv"
products_path = "C:/Users/hp/Downloads/archive (1)/product_details.csv"
sales_path = "C:/Users/hp/Downloads/archive (1)/E-commerece sales data 2024.csv"

# ================================================
# APP CONFIG
# ================================================
st.set_page_config(page_title='Sales Analytics Web', layout='wide')

# ================================================
# DATA LOADING AND MERGING
# ================================================
@st.cache_data
def load_data():
    customers = pd.read_csv(customers_path)
    products = pd.read_csv(products_path)
    sales = pd.read_csv(sales_path)

    # Clean column names
    for df in (customers, products, sales):
        df.columns = (
            df.columns
            .str.strip()
            .str.replace(r"[\s\-()/]", "_", regex=True)
            .str.replace(r"__+", "_", regex=True)
            .str.strip("_")
        )

    # Merge datasets
    df = (
        sales
        .merge(customers, on='Customer_ID', how='left')
        .merge(products, on='Product_ID', how='left')
    )
    return df

# ================================================
# PREPROCESSING & FEATURE ENGINEERING
# ================================================
def preprocess(df):
    df = df.copy()

    # Clean numeric columns
    numeric_cols = [
        'Age', 'Previous_Purchases', 'List_Price',
        'Selling_Price', 'Shipping_Weight', 'Purchase_Amount_USD'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            # Remove non-numeric characters and convert to float
            df[col] = df[col].astype(str).str.replace('[^\d.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle missing values
    if 'Interaction_type' in df.columns:
        df['Interaction_type'] = df['Interaction_type'].fillna('unknown')

    # Create transaction value
    if 'Purchase_Amount_USD' in df.columns and 'Previous_Purchases' in df.columns:
        df['Transaction_Value'] = df['Purchase_Amount_USD'] * df['Previous_Purchases']

    # Process timestamps
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.dropna(subset=['Timestamp'])
        df['Year'] = df['Timestamp'].dt.year
        df['Month'] = df['Timestamp'].dt.month
        df['Week'] = df['Timestamp'].dt.isocalendar().week
        df['Quarter'] = df['Timestamp'].dt.quarter
        df['DayOfWeek'] = df['Timestamp'].dt.day_name()
        df['Hour'] = df['Timestamp'].dt.hour
        df['Customer_Lifetime'] = (datetime.now() - df['Timestamp']).dt.days

    # Final numeric imputation
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    return df

# ================================================
# MODEL TRAINING
# ================================================
def build_and_train(df, n_iter):
    df_valid = df.copy()
    df_valid = df_valid.dropna(subset=['Purchase_Amount_USD'])

    # Feature selection
    features = [c for c in [
        'Age', 'Previous_Purchases', 'List_Price', 'Selling_Price',
        'Shipping_Weight', 'Customer_Lifetime', 'Transaction_Value'
    ] if c in df_valid.columns]

    # Ensure numeric types
    df_valid[features] = df_valid[features].apply(pd.to_numeric, errors='coerce')
    df_valid[features] = df_valid[features].fillna(df_valid[features].median())

    # Split data
    X = df_valid[features]
    y = df_valid['Purchase_Amount_USD']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline setup
    pipe = Pipeline([
        ('scale', ColumnTransformer([('num', StandardScaler(), features)], remainder='passthrough')),
        ('model', RandomForestRegressor(random_state=42))
    ])

    # Hyperparameter tuning
    param_dist = {
        'model__n_estimators': stats.randint(100, 500),
        'model__max_depth': stats.randint(5, 30),
        'model__max_features': ['sqrt', 'log2']  # removed 'auto' to avoid invalid parameter errors
    }
    
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=3,
        scoring='neg_root_mean_squared_error',
        random_state=42,
        n_jobs=-1  # use all processors for faster training
    )
    search.fit(X_train, y_train)

    # Evaluation
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    #compute RMSE manually since sklearn version doesn't support `squared` arg
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    return best_model, X_test, y_test, y_pred, {'RMSE': rmse, 'MAE': mae}, search.best_params_

# ================================================
# MAIN APP
# ================================================
def main():
    df_raw = load_data()
    if df_raw.empty:
        st.error("No data loaded – check your file paths!")
        return

    df = preprocess(df_raw)

    st.title('📊 E-commerce Sales Analytics')
    choice = option_menu(
        None,
        ['Overview', 'Trends', 'Customers', 'Predict'],
        icons=['house', 'graph-up', 'people', 'cpu'],
        orientation='horizontal'
    )

    # ---- Overview Tab ----
    if choice == 'Overview':
        st.subheader('Data Snapshot')
        st.dataframe(df.head(), use_container_width=True)
        
        st.subheader('Data Health Check')
        mv = df.isna().mean().sort_values(ascending=False).head(10)
        fig = px.bar(x=mv.index, y=mv.values, labels={'x': 'Column', 'y': '% Missing'})
        st.plotly_chart(fig, use_container_width=True)

    # ---- Trends ----
    elif choice == 'Trends':
        st.subheader('📈 Sales Over Time')

        if 'Interaction_type' in df.columns:
            interaction_count = df['Interaction_type'].value_counts()
            fig_interaction = px.pie(
                values=interaction_count.values,
                names=interaction_count.index,
                title='Buy View & Unknown Interaction Types'
            )
            st.plotly_chart(fig_interaction, use_container_width=True)

        if 'Week' in df.columns:
            weekly = df.groupby('Week')['Purchase_Amount_USD'].sum().reset_index()
            fig1 = px.line(weekly, x='Week', y='Purchase_Amount_USD', title='Weekly Sales Trend')
            st.plotly_chart(fig1, use_container_width=True)

        if 'Month' in df.columns:
            monthly = df.groupby('Month')['Purchase_Amount_USD'].sum().reset_index()
            fig2 = px.bar(monthly, x='Month', y='Purchase_Amount_USD', title='Monthly Sales Trend')
            st.plotly_chart(fig2, use_container_width=True)

        if 'Quarter' in df.columns:
            quarterly = df.groupby('Quarter')['Purchase_Amount_USD'].sum().reset_index()
            fig3 = px.bar(quarterly, x='Quarter', y='Purchase_Amount_USD', title='Quarterly Sales Trend')
            st.plotly_chart(fig3, use_container_width=True)

        if 'Product_Name' in df.columns:
            st.subheader('Top Products by Revenue')
            prod_rev = (
                df.dropna(subset=['Product_Name', 'Purchase_Amount_USD'])
                  .groupby('Product_Name')['Purchase_Amount_USD']
                  .sum().nlargest(10).reset_index()
            )
            fig5 = px.bar(prod_rev, x='Purchase_Amount_USD', y='Product_Name', orientation='h',
                         title='Top 10 Products by Revenue')
            st.plotly_chart(fig5, use_container_width=True)

        if 'Location' in df.columns:
            st.subheader('Top Locations by Revenue')
            top_locations = (
                df.dropna(subset=['Location'])
                  .groupby('Location')['Purchase_Amount_USD']
                  .sum().nlargest(10).reset_index()
            )
            fig_top_locations = px.bar(top_locations, x='Purchase_Amount_USD', y='Location', orientation='h',
                                       title='Top 10 Locations by Revenue')
            st.plotly_chart(fig_top_locations, use_container_width=True)

        if 'Month' in df.columns:
            avg_order_value = df.groupby('Month')['Purchase_Amount_USD'].mean().reset_index()
            fig7 = px.line(avg_order_value, x='Month', y='Purchase_Amount_USD', title='Average Order Value Over Time')
            st.plotly_chart(fig7, use_container_width=True)

        if 'DayOfWeek' in df.columns and 'Hour' in df.columns:
            sales_heatmap = df.groupby(['DayOfWeek', 'Hour'])['Purchase_Amount_USD'].sum().reset_index()
            fig8 = px.density_heatmap(sales_heatmap, x='DayOfWeek', y='Hour', z='Purchase_Amount_USD',
                                      title='Sales Heatmap (Day vs Hour)')
            st.plotly_chart(fig8, use_container_width=True)

    # ---- Customers Tab ----
    elif choice == 'Customers':
        st.subheader('👥 Customer Segmentation')
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Gender Distribution")
            gender_count = df['Gender'].value_counts()
            fig = px.pie(gender_count, values=gender_count.values, 
                        names=gender_count.index, hole=0.3)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Age Distribution")
            age_bins = [0, 18, 25, 35, 45, 55, 100]
            age_labels = ['<18', '18-24', '25-34', '35-44', '45-54', '55+']
            df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
            age_dist = df['Age Group'].value_counts().sort_index()
            fig = px.bar(age_dist, x=age_dist.index, y=age_dist.values,
                        title='Customer Age Distribution')
            st.plotly_chart(fig, use_container_width=True)

        st.subheader('📈 Customer Lifetime Value')
        clv = df.groupby('Customer_ID')['Purchase_Amount_USD'].sum().reset_index()
        fig = px.histogram(clv, x='Purchase_Amount_USD', 
                          nbins=50, title='Customer Lifetime Value Distribution')
        st.plotly_chart(fig, use_container_width=True)

    # ---- Predict Tab ----
    elif choice == 'Predict':
        st.subheader('🔮 Purchase Amount Prediction')
        
        st.markdown("### Model Configuration")
        n_iter = st.slider('Number of Hyperparameter Combinations to Try', 10, 100, 30)
        
        if st.button('Train Prediction Model'):
            with st.spinner('Training model... This may take a few minutes'):
                model, X_test, y_test, y_pred, metrics, params = build_and_train(df, n_iter)
            
            st.success("Model trained successfully!")
            st.markdown(f"""
            **Model Performance:**
            - RMSE: ${metrics['RMSE']:.2f}
            - MAE: ${metrics['MAE']:.2f}
            """ )
            
            st.markdown("### Prediction Visualization")
            fig = px.scatter(x=y_test, y=y_pred, 
                            labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                            title='Actual vs Predicted Values')
            fig.add_shape(type="line", x0=float(y_test.min()), y0=float(y_test.min()),
                          x1=float(y_test.max()), y1=float(y_test.max()),
                          line=dict(color="Red", width=2))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Best Model Parameters")
            st.json(params)

if __name__ == '__main__':
    main()
