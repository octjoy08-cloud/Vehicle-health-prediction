import pandas as pd
import numpy as np

def process_telemetry_batch(raw_data_list):
    """Flatten and clean telemetry JSON records."""
    records = []
    for batch in raw_data_list:
        if 'messages' in batch and isinstance(batch['messages'], list):
            records.extend(batch['messages'])
    
    df = pd.DataFrame(records)
    if 'eventDateTime' in df.columns:
        df['eventDateTime'] = pd.to_datetime(df['eventDateTime'], format='ISO8601')
    if 'headerData' in df.columns:
        df = df.drop(columns=['headerData'])
    
    df.drop_duplicates(inplace=True)
    return df

def calculate_vhi_and_status(df):
    """Compute Vehicle Health Index (VHI) and Health Status based on battery voltage."""
    if 'vehicleBatteryVoltage' not in df.columns:
        return df

    min_val = df['vehicleBatteryVoltage'].min()
    max_val = df['vehicleBatteryVoltage'].max()

    def calc_vhi(v):
        x = [min_val, 23.5, 24.0, 24.5, 25.0, max_val]
        y = [0.0, 10.0, 40.0, 70.0, 100.0, 100.0]
        return float(np.clip(np.interp(v, x, y), 0.0, 100.0))

    df['vhi'] = df['vehicleBatteryVoltage'].apply(calc_vhi)
    
    def status_label(vhi):
        if vhi >= 70:
            return 'Healthy'
        elif 40 <= vhi < 70:
            return 'Degrading'
        else:
            return 'Critical'

    df['health_status'] = df['vhi'].apply(status_label)
    return df
