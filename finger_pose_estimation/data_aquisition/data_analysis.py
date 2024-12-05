import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt
from sklearn.preprocessing import StandardScaler
import mne
import pickle
from sklearn.preprocessing import LabelEncoder

def build_emg_columns(df: pd.DataFrame):
    return [col for col in df.columns if 'channel' in col.lower()]


def build_leap_columns(full=False):
    #  if full build R21 else build R16
    fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    joints = ['TMC', 'MCP', 'PIP', 'DIP']
    movments = ['Flex', 'Adb']
    leap_columns = []
    if not full:
        #  remove DIP
        joints.remove('DIP')
    for finger in fingers:
        for joint in joints:
            for flex in movments:
                if finger != 'Thumb' and joint == 'TMC':
                    continue
                if finger != 'Thumb' and joint not in ['TMC', 'MCP'] and flex == 'Adb':
                    continue
                if finger == 'Thumb' and joint not in ['TMC', 'MCP', 'DIP']:
                    continue
                if finger == 'Thumb' and joint == 'DIP' and flex == 'Adb':
                    continue
                leap_columns.append(f'{finger}_{joint}_{flex}')
    return leap_columns

def read_emg(emg_path, fs=500):
    """Read EMG data from EDF file and convert to DataFrame"""
    raw = mne.io.read_raw_edf(emg_path, preload=True, verbose=False)
    emg_df = raw.to_data_frame()
    
    # Convert time column to datetime
    start_time = raw.info['meas_date']
    emg_df['time'] = pd.to_datetime(emg_df['time'], unit='s', origin=start_time)
    emg_df.set_index('time', inplace=True)
    
    return emg_df

def read_leap(leap_path):
    """Read LEAP motion data from CSV file"""
    leap_df = pd.read_csv(leap_path)
    leap_df['time'] = pd.to_datetime(leap_df['time'])
    leap_df = leap_df.set_index('time')
    return leap_df

def filter_emg(data, fs=500):
    """Apply filtering according to protocol 5.4.1"""
    # High-pass filter (20 Hz cutoff)
    sos = butter(4, 20, btype='highpass', output='sos', fs=fs)
    data = sosfiltfilt(sos, data, axis=0)
    
    # Notch filters for 50 Hz and 100 Hz
    for freq in [50, 100]:
        b_notch, a_notch = iirnotch(freq, 30, fs=fs)
        data = filtfilt(b_notch, a_notch, data, axis=0)
    
    return data

def segment_data(data, window_ms=512, stride_ms=2, sampling_rate=500):
    """Segment data using rolling window technique (5.5.1)"""
    window_samples = int(window_ms * (sampling_rate / 1000))  # Convert ms to samples (fs=500Hz)
    stride_samples = int(stride_ms * (sampling_rate / 1000))
    
    segments = []
    for i in range(0, len(data) - window_samples, stride_samples):
        segment = data[i:i + window_samples]
        segments.append(segment)
    
    return np.array(segments)


def spatial_grid_transform(segments):
    """Transform channels into 4x4 grid (5.5.4)"""
    # Assuming 16 channels in specific order
    grid_segments = segments.reshape(segments.shape[0], 4, 4, -1)
    return grid_segments

def merge_data(emg_data, leap_data):
    #  ajust the time
    start_time = max(min(emg_data.index), min(leap_data.index))
    end_time = min(max(emg_data.index), max(leap_data.index))

    emg_data = emg_data[start_time:end_time]
    leap_data = leap_data[start_time:end_time]

    data = pd.merge_asof(emg_data, leap_data, left_index=True, right_index=False, right_on='time', direction='backward', tolerance=pd.to_timedelta(10, unit='ms'))
    print(data.shape)
    data['gesture_class'] = data['gesture'].apply(lambda x: x.split('_')[0])
    # data['time_diff'] = (data.index - data['time_leap']).dt.total_seconds()
    # data.drop(columns=['timestamp', 'frame_id', 'time_leap'], inplace=True)


    #  reorder columns to have gesture at the end
    if 'gesture' in data.columns:
        data = data[[col for col in data.columns if col != 'gesture'] + ['gesture']]

    return data

def process_data(emg_path, leap_path, save_path, sampling_rate=500):
    """Main processing pipeline"""
    results = {}
    # Read data
    emg_data = read_emg(emg_path, fs=sampling_rate)
    leap_data = read_leap(leap_path)
    emg_channels = build_emg_columns(emg_data)
    leap_columns = build_leap_columns(full=False)

    results['emg_columns'] = emg_channels
    results['label_columns'] = leap_columns

    # Preprocessing
    emg_data = emg_data[emg_channels]
    emg_data[emg_channels] = filter_emg(emg_data, fs=sampling_rate)
    
    # Normalize
    scaler = StandardScaler()
    emg_data = scaler.fit_transform(emg_data)
        #  merge the data
    data = merge_data(emg_data, leap_data)
    # Filter
    
        # label encoder
    le = LabelEncoder()
    data['gesture'] = le.fit_transform(data['gesture'])
    results['gesture_mapping'] = le.classes_

    data['gesture_class'] = le.fit_transform(data['gesture_class'])
    results['gesture_mapping_class'] = le.classes_

    # Sort the data by the time index
    data = data.sort_index()

    # Store the original time range
    original_start_time = data.index.min()
    original_end_time = data.index.max()

    def remove_nan_sequences(data):
        # Function to remove leading and trailing NaN rows based on all columns
        def _remove_nan_sequences(df):
            # Drop 'gesture' column to focus on the other columns
            non_gesture_data = df.drop(columns=['gesture'])

            # Create a mask to identify rows with at least one non-NaN value
            mask = non_gesture_data.notna().any(axis=1)

            # Check if there are any valid rows in the group
            if mask.any():
                # Find the first and last index where any non-NaN value exists
                start_idx = mask.idxmax()
                end_idx = mask[::-1].idxmax()

                # Slice the dataframe to keep only the rows between start_idx and end_idx
                return df.loc[start_idx:end_idx]
            else:
                # If all rows are NaN, return an empty DataFrame
                return pd.DataFrame(columns=df.columns)

        # Process each group individually and concatenate the results
        grouped = data.groupby('gesture', sort=False)
        cleaned_data = pd.concat([_remove_nan_sequences(group) for _, group in grouped])

        # Reset index to clean up the final DataFrame
        cleaned_data = cleaned_data
        # Reset index to clean up the final DataFrame
        return cleaned_data
    # Sort the data again to ensure monotonically increasing index
    data = data.sort_index()
    # Validate the time index after processing
    if data.index.min() != original_start_time or data.index.max() != original_end_time:
        print(f"Warning: Time range has changed. Original: {original_start_time} to {original_end_time}")
        print(f"New: {data.index.min()} to {data.index.max()}")

    # Ensure the index is still monotonically increasing
    if not data.index.is_monotonic_increasing:
        raise ValueError("Time index is not monotonically increasing after processing.")

    # Add time information to the results
    results['start_time'] = data.index.min()
    results['end_time'] = data.index.max()
    results['time_step'] = data.index.to_series().diff().median()

    # Convert index to seconds from start for easier processing later
    # data['time_seconds'] = (data.index - data.index.min()).total_seconds()

    #  descritise the data
    # data = discritise_data(data)
    print(type(data))
    results['data'] = data#.values

    # Validate
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process EMG data")
    parser.add_argument('--emg_path', type=str, required=True, help='Path to EMG file')
    parser.add_argument('--leap_path', type=str, required=True, help='Path to LEAP motion file')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save processed data')
    parser.add_argument('--sampling_rate', type=int, default=500, help='Sampling rate in Hz (default: 500)')

    args = parser.parse_args()
    process_data(
        args.emg_path, 
        args.leap_path, 
        args.save_path, 
        args.sampling_rate,

    )
