import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

try:
    import joblib
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])
    import joblib


# Navigation
st.sidebar.title("Pages")
page = st.sidebar.radio("Go to", ["Item Qty Prediction", "Net Sales Prediction", "Item Qty Prediction 2", "Net Sales Prediction 2"])

if page == "Item Qty Prediction":
    class XGBoostModel:
        def load_model(self, model_path):
            # Load the model from the pickle file
            self.model_qty, self.label_encoder_store, self.label_encoder_dept = joblib.load(model_path)
        
        def predict(self, date, item_dept, store):

            testing_data = pd.read_csv("data/testing_model_data.csv")

            # Prepare the input data
            date_ordinal = pd.to_datetime(date).toordinal()
            item_dept_encoded = self.label_encoder_dept.transform([item_dept])[0]
            store_encoded = self.label_encoder_store.transform([store])[0]

            # Convert the date string to a datetime object and then get the previous date
            current_date = pd.to_datetime(date)
            previous_date = (current_date - pd.Timedelta(days=1)).strftime('%#Y-%#m-%#d')
            formatted_date = datetime.strptime(previous_date, '%Y-%m-%d').strftime('%Y-%m-%d')

            previous_two_date = (current_date - pd.Timedelta(days=2)).strftime('%#Y-%#m-%#d')
            formatted_two_date = datetime.strptime(previous_two_date, '%Y-%m-%d').strftime('%Y-%m-%d')

            # Retrieve the previous day's lag data from the test dataset
            prev_day_data = testing_data[(testing_data['date_id'] == formatted_date) & 
                                    (testing_data['item_dept'] == item_dept) & 
                                    (testing_data['store'] == store)]
            
            prev_two_day_data = testing_data[(testing_data['date_id'] == formatted_two_date) & 
                                    (testing_data['item_dept'] == item_dept) & 
                                    (testing_data['store'] == store)]
            
            if not prev_day_data.empty:
                item_qty_lag_1 = prev_day_data['item_qty'].values[0]
                net_sales_lag_1 = prev_day_data['net_sales'].values[0]
            else:
                # Default values if previous day data is not available
                item_qty_lag_1 = 0
                net_sales_lag_1 = 0

            if not prev_two_day_data.empty:
                item_qty_lag_2 = prev_two_day_data['item_qty'].values[0]
                net_sales_lag_2 = prev_two_day_data['net_sales'].values[0]
            else:
                # Default values if previous two days data is not available
                item_qty_lag_2 = 0
                net_sales_lag_2 = 0
            
            X_new = pd.DataFrame({
                'date_id': [date_ordinal],
                'item_dept': [item_dept_encoded],
                'store': [store_encoded],
                'item_qty_lag_1': [item_qty_lag_1],
                'net_sales_lag_1': [net_sales_lag_1],
                'item_qty_lag_2': [item_qty_lag_2],
                'net_sales_lag_2': [net_sales_lag_2]
            })
            
            # Predict the item_qty
            prediction = self.model_qty.predict(X_new)
            print(f"Predicted item_qty: {prediction[0]}")
            return prediction[0]
        
        def predict_february(self, item_dept, store):
            predictions = {}
            current_date = datetime.strptime("2022/02/01", "%Y/%m/%d")
            end_date = datetime.strptime("2022/02/28", "%Y/%m/%d")

            while current_date <= end_date:
                prediction = self.predict(current_date.strftime('%Y-%m-%d'), item_dept, store)
                predictions[current_date.strftime('%Y-%m-%d')] = prediction
                current_date += timedelta(days=1)

            return predictions

    # Initialize the model
    xgb_model = XGBoostModel()

    st.title("Item Quantity Prediction")
    date_input = st.date_input("Select Date", value=datetime.strptime("2022/02/01", "%Y/%m/%d").date())
    department_input = st.text_input("Enter Item Department", value="Beverages")
    store_input = st.text_input("Enter Store", value="ABC")
    if st.button("Predict Item Quantity"):
        xgb_model.load_model('src/models/xgb_model_qty.pkl')
        prediction = xgb_model.predict(str(date_input), department_input, store_input)
        st.write(f"Predicted Item Quantity: {prediction}")
    if st.button("Predict February"):
        xgb_model.load_model('src/models/xgb_model_qty.pkl')
        february_predictions = xgb_model.predict_february(department_input, store_input)
        for date, prediction in february_predictions.items():
            st.write(f"{date}: {prediction}")

elif page == "Net Sales Prediction":
    class XGBoostModel:
        def load_model_sales(self, model_path):
            # Load the model from the pickle file
            self.model_sales, self.label_encoder_store, self.label_encoder_dept = joblib.load(model_path)
        
        def predict_sales(self, date, item_dept, store):

            testing_data = pd.read_csv("data/testing_model_data.csv")

            # Prepare the input data
            date_ordinal = pd.to_datetime(date).toordinal()
            item_dept_encoded = self.label_encoder_dept.transform([item_dept])[0]
            store_encoded = self.label_encoder_store.transform([store])[0]

            # Convert the date string to a datetime object and then get the previous date
            current_date = pd.to_datetime(date)
            previous_date = (current_date - pd.Timedelta(days=1)).strftime('%#Y-%#m-%#d')
            formatted_date = datetime.strptime(previous_date, '%Y-%m-%d').strftime('%Y-%m-%d')

            previous_two_date = (current_date - pd.Timedelta(days=2)).strftime('%#Y-%#m-%#d')
            formatted_two_date = datetime.strptime(previous_two_date, '%Y-%m-%d').strftime('%Y-%m-%d')

            # Retrieve the previous day's lag data from the test dataset
            prev_day_data = testing_data[(testing_data['date_id'] == formatted_date) & 
                                    (testing_data['item_dept'] == item_dept) & 
                                    (testing_data['store'] == store)]
            
            prev_two_day_data = testing_data[(testing_data['date_id'] == formatted_two_date) & 
                                    (testing_data['item_dept'] == item_dept) & 
                                    (testing_data['store'] == store)]
            
            if not prev_day_data.empty:
                item_qty_lag_1 = prev_day_data['item_qty'].values[0]
                net_sales_lag_1 = prev_day_data['net_sales'].values[0]
            else:
                # Default values if previous day data is not available
                item_qty_lag_1 = 0
                net_sales_lag_1 = 0

            if not prev_two_day_data.empty:
                item_qty_lag_2 = prev_two_day_data['item_qty'].values[0]
                net_sales_lag_2 = prev_two_day_data['net_sales'].values[0]
            else:
                # Default values if previous two days data is not available
                item_qty_lag_2 = 0
                net_sales_lag_2 = 0
            
            X_new = pd.DataFrame({
                'date_id': [date_ordinal],
                'item_dept': [item_dept_encoded],
                'store': [store_encoded],
                'item_qty_lag_1': [item_qty_lag_1],
                'net_sales_lag_1': [net_sales_lag_1],
                'item_qty_lag_2': [item_qty_lag_2],
                'net_sales_lag_2': [net_sales_lag_2]
            })
            
            # Predict the net_sales
            prediction = self.model_sales.predict(X_new)
            print(f"Predicted net_sales: {prediction[0]}")
            return prediction[0]
        
        def predict_february(self, item_dept, store):
            predictions = {}
            current_date = datetime.strptime("2022/02/01", "%Y/%m/%d")
            end_date = datetime.strptime("2022/02/28", "%Y/%m/%d")

            while current_date <= end_date:
                prediction = self.predict_sales(current_date.strftime('%Y-%m-%d'), item_dept, store)
                predictions[current_date.strftime('%Y-%m-%d')] = prediction
                current_date += timedelta(days=1)

            return predictions

    # Initialize the model
    xgb_model = XGBoostModel()

    st.title("Net Sales Prediction")
    date_input = st.date_input("Select Date", value=datetime.strptime("2022/02/01", "%Y/%m/%d").date())
    department_input = st.text_input("Enter Item Department", value="Beverages")
    store_input = st.text_input("Enter Store", value="ABC")
    if st.button("Predict Net Sales"):
        xgb_model.load_model_sales('src/models/xgb_model_sales.pkl')
        prediction = xgb_model.predict_sales(str(date_input), department_input, store_input)
        st.write(f"Predicted Net Sales: {prediction}")
    if st.button("Predict February"):
        xgb_model.load_model_sales('src/models/xgb_model_sales.pkl')
        february_predictions = xgb_model.predict_february(department_input, store_input)
        for date, prediction in february_predictions.items():
            st.write(f"{date}: {prediction}")

elif page == "Item Qty Prediction 2":
    class XGBoostModel:
        def load_model_sales(self, model_path):
            # Load the model from the pickle file
            self.model_sales, self.label_encoder_store, self.label_encoder_dept = joblib.load(model_path)
        
        def predict_sales(self, date, store):

            testing_data = pd.read_csv("data/testing_model_data.csv")

            # Prepare the input data
            date_ordinal = pd.to_datetime(date).toordinal()
            store_encoded = self.label_encoder_store.transform([store])[0]

            # Convert the date string to a datetime object and then get the previous date
            current_date = pd.to_datetime(date)
            previous_date = (current_date - pd.Timedelta(days=1)).strftime('%#Y-%#m-%#d')
            formatted_date = datetime.strptime(previous_date, '%Y-%m-%d').strftime('%Y-%m-%d')

            previous_two_date = (current_date - pd.Timedelta(days=2)).strftime('%#Y-%#m-%#d')
            formatted_two_date = datetime.strptime(previous_two_date, '%Y-%m-%d').strftime('%Y-%m-%d')

            # Retrieve the previous day's lag data from the test dataset
            prev_day_data = testing_data[(testing_data['date_id'] == formatted_date) & 
                                    (testing_data['store'] == store)]
            
            prev_two_day_data = testing_data[(testing_data['date_id'] == formatted_two_date) & 
                                    (testing_data['store'] == store)]
            
            if not prev_day_data.empty:
                item_qty_lag_1 = prev_day_data['item_qty'].values[0]
                net_sales_lag_1 = prev_day_data['net_sales'].values[0]
            else:
                # Default values if previous day data is not available
                item_qty_lag_1 = 0
                net_sales_lag_1 = 0

            if not prev_two_day_data.empty:
                item_qty_lag_2 = prev_two_day_data['item_qty'].values[0]
                net_sales_lag_2 = prev_two_day_data['net_sales'].values[0]
            else:
                # Default values if previous two days data is not available
                item_qty_lag_2 = 0
                net_sales_lag_2 = 0
            
            X_new = pd.DataFrame({
                'date_id': [date_ordinal],
                'store': [store_encoded],
                'item_qty_lag_1': [item_qty_lag_1],
                'net_sales_lag_1': [net_sales_lag_1],
                'item_qty_lag_2': [item_qty_lag_2],
                'net_sales_lag_2': [net_sales_lag_2]
            })
            
            # Predict the net_sales
            prediction = self.model_sales.predict(X_new)
            print(f"Predicted net_sales: {prediction[0]}")
            return prediction[0]
        
        def predict_sales_february(self, store):
            # Predict sales for each day in February
            predictions = {}
            current_date = datetime.strptime("2022/02/01", "%Y/%m/%d")
            end_date = datetime.strptime("2022/02/28", "%Y/%m/%d")

            while current_date <= end_date:
                prediction = self.predict_sales(current_date.strftime('%Y-%m-%d'), store)
                predictions[current_date.strftime('%Y-%m-%d')] = prediction
                current_date += timedelta(days=1)

            return predictions

    # Initialize the model
    xgb_model = XGBoostModel()

    st.title("Item Qty Prediction Without Item Department")
    date_input = st.date_input("Select Date", value=datetime.strptime("2022/02/01", "%Y/%m/%d").date())
    store_input = st.text_input("Enter Store", value="ABC")
    if st.button("Predict Item Quantity"):
        xgb_model.load_model_sales('src/models/xgb_model_qty_wdepartment.pkl')
        prediction = xgb_model.predict_sales(str(date_input), store_input)
        st.write(f"Predicted Item Quantity: {prediction}")
    if st.button("Predict February Item Quantity"):
        xgb_model.load_model_sales('src/models/xgb_model_qty_wdepartment.pkl')
        february_predictions = xgb_model.predict_sales_february(store_input)
        for date, prediction in february_predictions.items():
            st.write(f"{date}: {prediction}")

elif page == "Net Sales Prediction 2":
    class XGBoostModel:
        def load_model_sales(self, model_path):
            # Load the model from the pickle file
            self.model_sales, self.label_encoder_store, self.label_encoder_dept = joblib.load(model_path)
        
        def predict_sales(self, date, store):

            testing_data = pd.read_csv("data/testing_model_data.csv")

            # Prepare the input data
            date_ordinal = pd.to_datetime(date).toordinal()
            store_encoded = self.label_encoder_store.transform([store])[0]

            # Convert the date string to a datetime object and then get the previous date
            current_date = pd.to_datetime(date)
            previous_date = (current_date - pd.Timedelta(days=1)).strftime('%#Y-%#m-%#d')
            formatted_date = datetime.strptime(previous_date, '%Y-%m-%d').strftime('%Y-%m-%d')

            previous_two_date = (current_date - pd.Timedelta(days=2)).strftime('%#Y-%#m-%#d')
            formatted_two_date = datetime.strptime(previous_two_date, '%Y-%m-%d').strftime('%Y-%m-%d')

            # Retrieve the previous day's lag data from the test dataset
            prev_day_data = testing_data[(testing_data['date_id'] == formatted_date) & 
                                    (testing_data['store'] == store)]
            
            prev_two_day_data = testing_data[(testing_data['date_id'] == formatted_two_date) & 
                                    (testing_data['store'] == store)]
            
            if not prev_day_data.empty:
                item_qty_lag_1 = prev_day_data['item_qty'].values[0]
                net_sales_lag_1 = prev_day_data['net_sales'].values[0]
            else:
                # Default values if previous day data is not available
                item_qty_lag_1 = 0
                net_sales_lag_1 = 0

            if not prev_two_day_data.empty:
                item_qty_lag_2 = prev_two_day_data['item_qty'].values[0]
                net_sales_lag_2 = prev_two_day_data['net_sales'].values[0]
            else:
                # Default values if previous two days data is not available
                item_qty_lag_2 = 0
                net_sales_lag_2 = 0
            
            X_new = pd.DataFrame({
                'date_id': [date_ordinal],
                'store': [store_encoded],
                'item_qty_lag_1': [item_qty_lag_1],
                'net_sales_lag_1': [net_sales_lag_1],
                'item_qty_lag_2': [item_qty_lag_2],
                'net_sales_lag_2': [net_sales_lag_2]
            })
            
            # Predict the net_sales
            prediction = self.model_sales.predict(X_new)
            print(f"Predicted net_sales: {prediction[0]}")
            return prediction[0]
        
        def predict_sales_february(self, store):
            # Predict sales for each day in February
            predictions = {}
            current_date = datetime.strptime("2022/02/01", "%Y/%m/%d")
            end_date = datetime.strptime("2022/02/28", "%Y/%m/%d")

            while current_date <= end_date:
                prediction = self.predict_sales(current_date.strftime('%Y-%m-%d'), store)
                predictions[current_date.strftime('%Y-%m-%d')] = prediction
                current_date += timedelta(days=1)

            return predictions

    # Initialize the model
    xgb_model = XGBoostModel()

    st.title("Net Sales Prediction")
    date_input = st.date_input("Select Date", value=datetime.strptime("2022/02/01", "%Y/%m/%d").date())
    store_input = st.text_input("Enter Store", value="ABC")
    if st.button("Predict Net Sales"):
        xgb_model.load_model_sales('src/models/xgb_model_sales_wdepartment.pkl')
        prediction = xgb_model.predict_sales(str(date_input), store_input)
        st.write(f"Predicted Net Sales: {prediction}")
    if st.button("Predict February Net Sales"):
        xgb_model.load_model_sales('src/models/xgb_model_sales_wdepartment.pkl')
        february_predictions = xgb_model.predict_sales_february(store_input)
        for date, prediction in february_predictions.items():
            st.write(f"{date}: {prediction}")