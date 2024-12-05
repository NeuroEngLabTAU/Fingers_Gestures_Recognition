import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt
from sklearn.preprocessing import StandardScaler
import mne
import pickle

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

def segment_data(data, window_ms=512, stride_ms=2):
    """Segment data using rolling window technique (5.5.1)"""
    window_samples = int(window_ms * 0.5)  # Convert ms to samples (fs=500Hz)
    stride_samples = int(stride_ms * 0.5)
    
    segments = []
    for i in range(0, len(data) - window_samples, stride_samples):
        segment = data[i:i + window_samples]
        segments.append(segment)
    
    return np.array(segments)

def validate_segments(emg_segments, leap_data):
    """Validate segments against HKD data (5.5.2, 5.5.3)"""
    valid_segments = []
    valid_labels = []
    
    for segment in emg_segments:
        # Add validation logic here based on HKD patterns
        # This is a placeholder for the validation logic
        pass
    
    return np.array(valid_segments), np.array(valid_labels)

def spatial_grid_transform(segments):
    """Transform channels into 4x4 grid (5.5.4)"""
    # Assuming 16 channels in specific order
    grid_segments = segments.reshape(segments.shape[0], 4, 4, -1)
    return grid_segments

def process_data(emg_path, leap_path, save_path, sampling_rate=500, window_ms=512, stride_ms=2):
    """Main processing pipeline"""
    # Read data
    emg_data = read_emg(emg_path, fs=sampling_rate)
    leap_data = read_leap(leap_path)
    
    # Preprocessing
    emg_channels = [col for col in emg_data.columns if 'channel' in col.lower()]
    emg_data = emg_data[emg_channels]
    
    # Normalize
    scaler = StandardScaler()
    emg_normalized = scaler.fit_transform(emg_data)
    
    # Filter
    emg_filtered = filter_emg(emg_normalized, fs=sampling_rate)
    
    # Segment
    emg_segments = segment_data(emg_filtered, window_ms=window_ms, stride_ms=stride_ms)
    
    # Validate
    valid_segments, valid_labels = validate_segments(emg_segments, leap_data)
    
    # Spatial transform
    grid_segments = spatial_grid_transform(valid_segments)
    
    # Save results
    results = {
        'segments': grid_segments,
        'labels': valid_labels,
        'scaler': scaler
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    
    return save_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process EMG data")
    parser.add_argument('--emg_path', type=str, required=True, help='Path to EMG file')
    parser.add_argument('--leap_path', type=str, required=True, help='Path to LEAP motion file')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save processed data')
    parser.add_argument('--sampling_rate', type=int, default=500, help='Sampling rate in Hz (default: 500)')
    parser.add_argument('--window_ms', type=int, default=512, help='Window size in milliseconds (default: 512)')
    parser.add_argument('--stride_ms', type=int, default=2, help='Stride interval in milliseconds (default: 2)')
    
    args = parser.parse_args()
    process_data(
        args.emg_path, 
        args.leap_path, 
        args.save_path, 
        args.sampling_rate,
        args.window_ms,
        args.stride_ms
    )