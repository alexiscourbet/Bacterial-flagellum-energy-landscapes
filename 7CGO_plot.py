#!/usr/bin/env python3

import pandas as pd
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse  # Added for command-line arguments

# Set up the visual style
sns.set(
    font_scale=1,
    style='darkgrid',
    palette='colorblind'
)
matplotlib.rcParams["figure.figsize"] = [12.0, 12.0]

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot energy scores vs angles with adjustable dot size.')
    parser.add_argument('--dot-size', type=float, default=50.0,
                      help='Size of dots in the plot. Use 0 for no dots, smaller values for smaller dots (default: 50.0)')
    parser.add_argument('--input-file', type=str, default="FD_score.sc",
                      help='Input score file to process (default: FD_score.sc)')
    return parser.parse_args()

def parse_score_file(filename):
    try:
        score_file = open(filename).readlines()
    except FileNotFoundError:
        print("{} not found!".format(filename))
        sys.exit(1)

    headers = score_file[1].split()[1:]
    score_data = [line.split()[1:] for line in score_file[2:]]
    return headers, score_data

def process_data(data):
    headers, score_data = parse_score_file(data)
    
    # Create DataFrame first without specifying dtype
    scores = pd.DataFrame(score_data, columns=headers)
    
    # Print the first few rows to see what's actually in the data
    print("First few rows of raw data:")
    if 'description' in scores.columns:
        print(scores['description'].head())
    else:
        print(scores.head())
    
    # Now convert columns that should be numeric
    for col in scores.columns:
        if col != 'description':
            try:
                scores[col] = pd.to_numeric(scores[col])
            except Exception as e:
                print(f"Cannot convert column {col} to numeric: {e}")
    
    scores.dropna(inplace=True)
    
    # Calculate score per residue
    scores['score_per_res'] = scores['total_score'] / 3784
    scores['%score_deviation'] = - (scores['score_per_res'] - scores['score_per_res'].mean()) / scores['score_per_res'].mean()
    
    # More robust extraction of rotation angle
    def extract_angle_from_description(desc):
        try:
            # Try different formats of descriptions
            parts = desc.split('_')
            
            # Debug
            print(f"Description parts: {parts}")
            
            # Look for a part that contains the word "angle"
            angle_part = None
            for part in parts:
                if 'angle' in part:
                    angle_part = part
                    break
            
            # If we found a part with "angle"
            if angle_part:
                # If the angle is in the format "angle_45.123"
                if angle_part == "angle":
                    # The angle may be in the next part
                    idx = parts.index(angle_part)
                    if idx + 1 < len(parts):
                        try:
                            return float(parts[idx + 1])
                        except ValueError:
                            pass
                # If the angle is in the format "angle45.123"
                else:
                    try:
                        angle_val = angle_part.replace('angle', '')
                        return float(angle_val)
                    except ValueError:
                        pass
            
            # If no part contains "angle", look for any part that could be a number
            for part in parts:
                try:
                    return float(part)
                except ValueError:
                    continue
            
            return None
            
        except Exception as e:
            print(f"Error extracting angle from {desc}: {e}")
            return None
    
    # Apply the extraction function to get angles
    scores['Rotation_angle'] = scores['description'].apply(extract_angle_from_description)
    
    # Print how many valid angles we extracted
    valid_angles = scores['Rotation_angle'].count()
    print(f"Found {valid_angles} valid angles out of {len(scores)} rows")
    
    # Drop rows with invalid angles
    scores = scores.dropna(subset=['Rotation_angle'])
    
    # Convert angles to float and radians
    scores['Rotation_angle_float'] = scores['Rotation_angle']
    scores['Rotation_angle_float_rad'] = scores['Rotation_angle_float'] * np.pi / 180
    
    # Create columns for aggregation
    scores['total_score_mean'] = scores['total_score']
    scores['total_score_sd'] = scores['total_score']
    scores['score_per_res_mean'] = scores['score_per_res']
    scores['score_per_res_sd'] = scores['score_per_res']
    scores['%score_deviation_mean'] = scores['%score_deviation']
    scores['%score_deviation_sd'] = scores['%score_deviation']
    
    # Group by rotation angle and calculate statistics
    scores2 = scores.groupby(['Rotation_angle_float']).agg({
        'total_score_mean': 'mean', 
        'total_score_sd': 'std',
        'score_per_res_mean': 'mean', 
        'score_per_res_sd': 'std',
        '%score_deviation_mean': 'mean', 
        '%score_deviation_sd': 'std'
    })
    
    # Print aggregated data to verify we have values
    print("Aggregated data:")
    print(scores2.head())
    
    scores2 = scores2.reset_index()
    scores2['Rotation_angle_float_rad'] = scores2['Rotation_angle_float'] * np.pi / 180
    sorted_scores = scores2.sort_values('Rotation_angle_float', ascending=True)
    
    # Final check that we have data for plotting
    print(f"Final data has {len(sorted_scores)} rows for plotting")
    
    return sorted_scores

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Process the data
    scores_7CGO = process_data(args.input_file)
    scores_7CGO.to_csv('scores_7CGO_centered.csv')
    
    # Print actual values that will be plotted
    print("\nData being plotted:")
    print(f"X values (angles in radians): {scores_7CGO['Rotation_angle_float_rad'].tolist()}")
    print(f"Y values (scores): {scores_7CGO['total_score_mean'].tolist()}")
    
    # Create the polar plot
    plt.figure(figsize=(12, 12))
    ax = plt.subplot(111, polar=True)
    
    # Check if we have data to plot
    if len(scores_7CGO) > 0:
        # Calculate y-axis limits: 10% below min and 10% above max
        if len(scores_7CGO['total_score_mean']) > 0:
            min_score = scores_7CGO['total_score_mean'].min()
            max_score = scores_7CGO['total_score_mean'].max()
            y_min = min_score - 0.035 * abs(min_score)
            y_max = max_score + 0.025 * abs(max_score)
            print(f"Setting y-axis limits from {y_min} to {y_max}")
            ax.set_ylim(y_min, y_max)
        
        ax.errorbar(
            scores_7CGO['Rotation_angle_float_rad'], 
            scores_7CGO['total_score_mean'], 
            yerr=scores_7CGO['total_score_sd'], 
            capsize=0, 
            color='red', 
            linewidth=2, 
            elinewidth=0.2, 
            alpha=0.7
        )
        
        # Add points to make data more visible (if dot size > 0)
        if args.dot_size > 0:
            ax.scatter(
                scores_7CGO['Rotation_angle_float_rad'], 
                scores_7CGO['total_score_mean'],
                color='blue',
                s=args.dot_size
            )
    else:
        print("WARNING: No data to plot!")
    
    ax.set_rlabel_position(90)
    ax.set_title("Energy score vs angle", va='bottom')
    
    # Save the figure
    plt.savefig('total_score_7CGO_centered.png', dpi=300, bbox_inches='tight')
    
    # Display plot info
    print("Plot saved to 'total_score_7CGO_centered.png'")
    print(f"Dots plotted with size: {args.dot_size}")
    
    # Create FFT analysis of the landscape
    if len(scores_7CGO) > 0:
        # Sort data by angle for FFT
        scores_sorted = scores_7CGO.sort_values('Rotation_angle_float')
        
        # Extract the energy values
        energy_values = scores_sorted['total_score_mean'].values
        
        # Remove mean to center the data (DC component)
        energy_values = energy_values - np.mean(energy_values)
        
        # Apply FFT
        fft_values = np.fft.rfft(energy_values)
        fft_amplitude = np.abs(fft_values)
        
        # Calculate frequencies
        n = len(energy_values)
        freq = np.fft.rfftfreq(n, d=1.0/n)  # Normalized frequency (cycles per data range)
        
        # Create FFT plot
        plt.figure(figsize=(10, 6))
        plt.plot(freq, fft_amplitude, 'r-', linewidth=2)
        plt.xlabel('Frequency')
        plt.ylabel('FFT Amplitude')
        plt.title('FFT Analysis of Energy Landscape')
        plt.grid(True)
        
        # Set x-axis limit to maximum 250
        plt.xlim(0, 250)
        
        # Add more visual distinction to dominant frequencies
        if len(fft_amplitude) > 0:
            # Find all local maxima (peaks)
            peaks = [i for i in range(1, len(fft_amplitude)-1) if 
                     fft_amplitude[i] > fft_amplitude[i-1] and 
                     fft_amplitude[i] > fft_amplitude[i+1]]
            
            # Sort peaks by amplitude (highest first) for better label placement
            peaks.sort(key=lambda i: fft_amplitude[i], reverse=True)
            
            # Highlight the most significant peaks with larger markers
            significant_peaks = [i for i in peaks if fft_amplitude[i] > 0.3 * np.max(fft_amplitude)]
            if significant_peaks:
                plt.scatter(freq[significant_peaks], fft_amplitude[significant_peaks], color='blue', s=60)
            
            # Highlight all other peaks with smaller markers
            minor_peaks = [i for i in peaks if i not in significant_peaks]
            if minor_peaks:
                plt.scatter(freq[minor_peaks], fft_amplitude[minor_peaks], color='green', s=30, alpha=0.7)
            
            # Label all peaks
            for i in peaks:
                if freq[i] <= 250:  # Only label peaks within our x-axis range
                    plt.annotate(f"{freq[i]:.1f}", 
                               (freq[i], fft_amplitude[i]),
                               textcoords="offset points", 
                               xytext=(0,10), 
                               ha='center',
                               fontsize=8)  # Smaller font size to avoid overcrowding
        
        plt.tight_layout()
        plt.savefig('fft_analysis_7CGO.png', dpi=300, bbox_inches='tight')
        print("FFT analysis plot saved to 'fft_analysis_7CGO.png'")

if __name__ == "__main__":
    main() 