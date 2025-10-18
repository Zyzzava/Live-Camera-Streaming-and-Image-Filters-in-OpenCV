import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create data directory structure
DATA_DIR = 'data'
PLOTS_DIR = os.path.join(DATA_DIR, 'plots')
TABLES_DIR = os.path.join(DATA_DIR, 'tables')

def setup_directories():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

def load_data(filename, data_type="general"):
    try:
        df = pd.read_csv(filename)
        
        if 'Resolution' in df.columns:
            resolutions = sorted([int(x) for x in df['Resolution'].unique() if pd.notna(x)])
            print(f"  Resolutions: {resolutions}")
        
        print(f"  Transform configs: {df['HasTransform'].unique().tolist()}")
        
        # Save raw data to data folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_data_path = os.path.join(DATA_DIR, f'raw_data_{data_type}_{timestamp}.csv')
        df.to_csv(raw_data_path, index=False)
        print(f"\nRaw data backed up to: {raw_data_path}")
        
        return df
    except FileNotFoundError:
        print(f"Error: {filename} not found!")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def analyze_transforms(filename='performance_transforms.csv'):
    df = load_data(filename, "transforms")
    plot_fps_comparison(df, "transforms")
    plot_transform_comparison(df)
    generate_summary_table(df, "transforms")

    return df

def analyze_resolutions(filename='performance_resolution.csv'):
    df = load_data(filename, "resolution")
    plot_resolution_impact(df)
    generate_summary_table(df, "resolution")
    generate_resolution_table(df)
    
    return df

def plot_fps_comparison(df, suffix=""):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # FPS by Filter and Execution Mode (No Transform)
    ax1 = axes[0, 0]
    df_no_transform = df[df['HasTransform'] == 'No']
    
    if len(df_no_transform) > 0:
        df_pivot = df_no_transform.pivot_table(
            values='FPS', 
            index='FilterName', 
            columns='ExecutionMode', 
            aggfunc='mean'
        )
        df_pivot.plot(kind='bar', ax=ax1)
        ax1.set_title('Average FPS by Filter (No Transform)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Filter', fontsize=12)
        ax1.set_ylabel('FPS', fontsize=12)
        ax1.legend(title='Execution Mode')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
    else:
        ax1.text(0.5, 0.5, 'No data without transforms', 
                ha='center', va='center', transform=ax1.transAxes)
    
    # Frame Time Comparison
    ax2 = axes[0, 1]
    if len(df_no_transform) > 0:
        df_pivot_time = df_no_transform.pivot_table(
            values='FrameTime_ms', 
            index='FilterName', 
            columns='ExecutionMode', 
            aggfunc='mean'
        )
        df_pivot_time.plot(kind='bar', ax=ax2, color=['#ff7f0e', '#2ca02c'])
        ax2.set_title('Average Frame Time by Filter (No Transform)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Filter', fontsize=12)
        ax2.set_ylabel('Frame Time (ms)', fontsize=12)
        ax2.legend(title='Execution Mode')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
    else:
        ax2.text(0.5, 0.5, 'No data without transforms', 
                ha='center', va='center', transform=ax2.transAxes)
    
    # GPU vs CPU Speedup
    ax3 = axes[1, 0]
    if len(df_no_transform) > 0:
        df_pivot_speedup = df_no_transform.pivot_table(
            values='FPS', 
            index='FilterName', 
            columns='ExecutionMode', 
            aggfunc='mean'
        )
        
        if 'CPU' in df_pivot_speedup.columns and 'GPU' in df_pivot_speedup.columns:
            speedup = df_pivot_speedup['GPU'] / df_pivot_speedup['CPU']
            speedup.plot(kind='bar', ax=ax3, color='#d62728')
            ax3.axhline(y=1, color='black', linestyle='--', linewidth=2, label='No speedup')
            ax3.set_title('GPU Speedup Factor (GPU FPS / CPU FPS)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Filter', fontsize=12)
            ax3.set_ylabel('Speedup Factor', fontsize=12)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='x', rotation=45)
            
            # Save speedup data to CSV
            speedup_df = pd.DataFrame({
                'Filter': speedup.index,
                'Speedup_Factor': speedup.values
            })
            speedup_path = os.path.join(TABLES_DIR, f'gpu_speedup_{suffix}.csv')
            speedup_df.to_csv(speedup_path, index=False)
            print(f"Saved speedup data: {speedup_path}")
        else:
            ax3.text(0.5, 0.5, 'Need both CPU and GPU data', 
                    ha='center', va='center', transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, 'No data available', 
                ha='center', va='center', transform=ax3.transAxes)
    
    # Transform Impact
    ax4 = axes[1, 1]
    if len(df['HasTransform'].unique()) > 1:
        df_transform = df.groupby(['ExecutionMode', 'HasTransform'])['FPS'].mean().unstack()
        df_transform.plot(kind='bar', ax=ax4)
        ax4.set_title('Transform Impact on FPS', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Execution Mode', fontsize=12)
        ax4.set_ylabel('Average FPS', fontsize=12)
        ax4.legend(title='Has Transform', labels=['No', 'Yes'])
        ax4.grid(True, alpha=0.3)
        
        # Save transform impact data
        transform_path = os.path.join(TABLES_DIR, f'transform_impact_{suffix}.csv')
        df_transform.to_csv(transform_path)
        print(f"Saved transform impact: {transform_path}")
    else:
        mode_avg = df.groupby('ExecutionMode')['FPS'].mean()
        mode_avg.plot(kind='bar', ax=ax4, color=['#1f77b4', '#ff7f0e'])
        transform_status = 'With' if df['HasTransform'].iloc[0] == 'Yes' else 'Without'
        ax4.set_title(f'Average FPS by Mode ({transform_status} Transform)', 
                     fontsize=14, fontweight='bold')
        ax4.set_xlabel('Execution Mode', fontsize=12)
        ax4.set_ylabel('FPS', fontsize=12)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, f'performance_comparison_{suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.show()

def plot_transform_comparison(df):
    if len(df['HasTransform'].unique()) <= 1:
        print("Skipping transform comparison (only one transform config)")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # FPS comparison with/without transform
    ax1 = axes[0]
    width = 0.35
    filters = df['FilterName'].unique()
    x = np.arange(len(filters))
    
    for mode_idx, mode in enumerate(['GPU', 'CPU']):
        df_mode = df[df['ExecutionMode'] == mode]
        
        no_transform = []
        with_transform = []
        
        for filt in filters:
            df_filter = df_mode[df_mode['FilterName'] == filt]
            no_t = df_filter[df_filter['HasTransform'] == 'No']['FPS'].mean()
            with_t = df_filter[df_filter['HasTransform'] == 'Yes']['FPS'].mean()
            no_transform.append(no_t if not pd.isna(no_t) else 0)
            with_transform.append(with_t if not pd.isna(with_t) else 0)
        
        offset = width * (mode_idx * 2 - 1) / 2
        ax1.bar(x + offset, no_transform, width/2, label=f'{mode} - No Transform', alpha=0.8)
        ax1.bar(x + offset + width/2, with_transform, width/2, label=f'{mode} - With Transform', alpha=0.6)
    
    ax1.set_title('FPS: Transform Impact by Filter and Mode', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Filter', fontsize=12)
    ax1.set_ylabel('FPS', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(filters, rotation=45, ha='right')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Transform overhead percentage
    ax2 = axes[1]
    overhead_data = []
    
    for mode in ['GPU', 'CPU']:
        df_mode = df[df['ExecutionMode'] == mode]
        
        for filt in filters:
            df_filter = df_mode[df_mode['FilterName'] == filt]
            no_transform_fps = df_filter[df_filter['HasTransform'] == 'No']['FPS'].mean()
            with_transform_fps = df_filter[df_filter['HasTransform'] == 'Yes']['FPS'].mean()
            
            if not pd.isna(no_transform_fps) and not pd.isna(with_transform_fps) and no_transform_fps > 0:
                overhead = ((no_transform_fps - with_transform_fps) / no_transform_fps * 100)
                overhead_data.append({
                    'Filter': filt,
                    'Mode': mode,
                    'Overhead': overhead
                })
    
    if overhead_data:
        overhead_df = pd.DataFrame(overhead_data)
        overhead_pivot = overhead_df.pivot(index='Filter', columns='Mode', values='Overhead')
        overhead_pivot.plot(kind='bar', ax=ax2, alpha=0.8)
        ax2.set_title('Transform Performance Overhead (%)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Filter', fontsize=12)
        ax2.set_ylabel('Performance Loss (%)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=45)
        
        # Save overhead data
        overhead_path = os.path.join(TABLES_DIR, 'transform_overhead.csv')
        overhead_pivot.to_csv(overhead_path)
        print(f"Saved overhead data: {overhead_path}")
    
    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, 'transform_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.show()

def plot_resolution_impact(df):
    if len(df['Resolution'].unique()) <= 1:
        print("Skipping resolution impact (only one resolution)")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # FPS vs Resolution by Mode
    ax1 = axes[0, 0]
    resolution_data = []
    
    for mode in df['ExecutionMode'].unique():
        df_mode = df[df['ExecutionMode'] == mode]
        resolution_avg = df_mode.groupby('Resolution')['FPS'].mean().sort_index()
        ax1.plot(resolution_avg.index, resolution_avg.values, 
                marker='o', linewidth=2, markersize=8, label=mode)
        
        for res, fps in zip(resolution_avg.index, resolution_avg.values):
            resolution_data.append({'Resolution': res, 'Mode': mode, 'FPS': fps})
    
    ax1.set_title('Average FPS vs Resolution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Resolution (pixels)', fontsize=12)
    ax1.set_ylabel('FPS', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # resolution labels
    res_labels = {
        640*480: 'VGA\n640x480',
        1280*720: 'HD\n1280x720',
        1920*1080: 'Full HD\n1920x1080'
    }
    unique_res = sorted(df['Resolution'].unique())
    ax1.set_xticks(unique_res)
    ax1.set_xticklabels([res_labels.get(r, f'{r}') for r in unique_res])
    
    # Frame Time vs Resolution
    ax2 = axes[0, 1]
    for mode in df['ExecutionMode'].unique():
        df_mode = df[df['ExecutionMode'] == mode]
        resolution_avg = df_mode.groupby('Resolution')['FrameTime_ms'].mean().sort_index()
        ax2.plot(resolution_avg.index, resolution_avg.values, 
                marker='s', linewidth=2, markersize=8, label=mode)
    
    ax2.set_title('Average Frame Time vs Resolution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Resolution (pixels)', fontsize=12)
    ax2.set_ylabel('Frame Time (ms)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(unique_res)
    ax2.set_xticklabels([res_labels.get(r, f'{r}') for r in unique_res])
    
    # Filter performance across resolutions (GPU)
    ax3 = axes[1, 0]
    df_gpu = df[df['ExecutionMode'] == 'GPU']
    
    for filt in df_gpu['FilterName'].unique():
        df_filter = df_gpu[df_gpu['FilterName'] == filt]
        res_fps = df_filter.groupby('Resolution')['FPS'].mean().sort_index()
        ax3.plot(res_fps.index, res_fps.values, 
                marker='o', linewidth=2, label=filt, alpha=0.7)
    
    ax3.set_title('GPU Filter Performance vs Resolution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Resolution (pixels)', fontsize=12)
    ax3.set_ylabel('FPS', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(unique_res)
    ax3.set_xticklabels([res_labels.get(r, f'{r}') for r in unique_res])
    
    # Resolution scaling efficiency
    ax4 = axes[1, 1]
    
    # Calculate pixels per second for each mode/resolution
    for mode in df['ExecutionMode'].unique():
        df_mode = df[df['ExecutionMode'] == mode]
        
        pixels_per_sec = []
        resolutions = []
        
        for res in sorted(df_mode['Resolution'].unique()):
            df_res = df_mode[df_mode['Resolution'] == res]
            avg_fps = df_res['FPS'].mean()
            pixels_per_sec.append(res * avg_fps / 1e6)  # Megapixels per second
            resolutions.append(res)
        
        ax4.plot(resolutions, pixels_per_sec, 
                marker='d', linewidth=2, markersize=8, label=mode)
    
    ax4.set_title('Processing Throughput (Megapixels/sec)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Resolution (pixels)', fontsize=12)
    ax4.set_ylabel('Megapixels/sec', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(unique_res)
    ax4.set_xticklabels([res_labels.get(r, f'{r}') for r in unique_res])
    
    # Save resolution data
    if resolution_data:
        res_df = pd.DataFrame(resolution_data)
        res_path = os.path.join(TABLES_DIR, 'resolution_impact.csv')
        res_df.to_csv(res_path, index=False)
        print(f"Saved resolution data: {res_path}")
    
    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, 'resolution_impact.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.show()

def generate_summary_table(df, suffix=""):
    
    # Overall statistics
    summary = df.groupby(['ExecutionMode', 'FilterName']).agg({
        'FPS': ['mean', 'std', 'min', 'max'],
        'FrameTime_ms': ['mean', 'std']
    }).round(2)
    
    # Save to CSV
    summary_path = os.path.join(TABLES_DIR, f'detailed_statistics_{suffix}.csv')
    summary.to_csv(summary_path)
    print(f"\nSaved detailed statistics: {summary_path}")
    
    # GPU vs CPU comparison
    mode_comparison = df.groupby(['ExecutionMode', 'HasTransform'])[['FPS', 'FrameTime_ms']].mean()
    
    mode_path = os.path.join(TABLES_DIR, f'mode_comparison_{suffix}.csv')
    mode_comparison.to_csv(mode_path)
    print(f"Saved mode comparison: {mode_path}")
    
    # Best and worst performers
    top5 = df.nlargest(5, 'FPS')[['FilterName', 'ExecutionMode', 'HasTransform', 'FPS', 'FrameTime_ms']]
    top5_path = os.path.join(TABLES_DIR, f'top_performers_{suffix}.csv')
    top5.to_csv(top5_path, index=False)
    print(f"Saved top performers: {top5_path}")
    
    bottom5 = df.nsmallest(5, 'FPS')[['FilterName', 'ExecutionMode', 'HasTransform', 'FPS', 'FrameTime_ms']]
    
    bottom5_path = os.path.join(TABLES_DIR, f'bottom_performers_{suffix}.csv')
    bottom5.to_csv(bottom5_path, index=False)
    print(f"Saved bottom performers: {bottom5_path}")

def generate_resolution_table(df):
    if len(df['Resolution'].unique()) <= 1:
        return
    
    # res name mapping
    res_names = {
        640*480: 'VGA (640x480)',
        1280*720: 'HD (1280x720)',
        1920*1080: 'Full HD (1920x1080)'
    }
    
    # Summary by res and mode
    res_summary = df.groupby(['Resolution', 'ExecutionMode']).agg({
        'FPS': ['mean', 'std', 'min', 'max'],
        'FrameTime_ms': ['mean']
    }).round(2)
    
    # Save to CSV
    res_table_path = os.path.join(TABLES_DIR, 'resolution_summary.csv')
    res_summary.to_csv(res_table_path)
    print(f"\nSaved resolution summary: {res_table_path}")

def generate_combined_report(transform_df, resolution_df):
    report_path = os.path.join(DATA_DIR, 'performance_report.txt')
    
    with open(report_path, 'w') as f:
        # Transform benchmark summary
        if transform_df is not None:
            f.write(f"Total measurements: {len(transform_df)}\n")
            
            filters = [str(x) for x in transform_df['FilterName'].unique() if pd.notna(x)]
            modes = [str(x) for x in transform_df['ExecutionMode'].unique() if pd.notna(x)]
            transforms = [str(x) for x in transform_df['HasTransform'].unique() if pd.notna(x)]
            
            f.write(f"Filters tested: {', '.join(filters)}\n")
            f.write(f"Execution modes: {', '.join(modes)}\n")
            f.write(f"Transform configurations: {', '.join(transforms)}\n\n")
            
            # Overall averages for transforms
            overall = transform_df.groupby('ExecutionMode')[['FPS', 'FrameTime_ms']].mean()
            f.write(overall.to_string())
            f.write("\n\n")
            
            # GPU speedup
            if 'GPU' in transform_df['ExecutionMode'].unique() and 'CPU' in transform_df['ExecutionMode'].unique():
                gpu_avg = transform_df[transform_df['ExecutionMode'] == 'GPU']['FPS'].mean()
                cpu_avg = transform_df[transform_df['ExecutionMode'] == 'CPU']['FPS'].mean()
                speedup = gpu_avg / cpu_avg
                f.write(f"Average GPU Speedup: {speedup:.2f}x\n\n")
        
        # Resolution benchmark summary
        if resolution_df is not None:
            res_names = {
                640*480: 'VGA (640x480)',
                1280*720: 'HD (1280x720)',
                1920*1080: 'Full HD (1920x1080)'
            }
            
            resolutions = [res_names.get(int(r), f'{int(r)} pixels') 
                          for r in resolution_df['Resolution'].unique() if pd.notna(r)]
            f.write(f"Resolutions tested: {', '.join(resolutions)}\n\n")
            
            for mode in resolution_df['ExecutionMode'].unique():
                if pd.notna(mode):
                    df_mode = resolution_df[resolution_df['ExecutionMode'] == mode]
                    f.write(f"\n{mode} Mode:\n")
                    
                    for res in sorted(df_mode['Resolution'].unique()):
                        if pd.notna(res):
                            df_res = df_mode[df_mode['Resolution'] == res]
                            avg_fps = df_res['FPS'].mean()
                            res_name = res_names.get(int(res), f'{int(res)} pixels')
                            f.write(f"  {res_name}: {avg_fps:.2f} FPS\n")

def main():
    # Setup directories
    setup_directories()
    # Analyze both datasets
    transform_df = analyze_transforms()
    resolution_df = analyze_resolutions()
    # Generate combined report if we have any data
    generate_combined_report(transform_df, resolution_df)

if __name__ == "__main__":
    main()