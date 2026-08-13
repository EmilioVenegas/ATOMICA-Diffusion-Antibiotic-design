import glob
import pandas as pd
import re

def parse_logs():
    log_files = glob.glob('docking/*_log.txt')
    results = []

    for log_file in log_files:
        ligand_name = log_file.replace('docking/', '').replace('_log.txt', '')
        best_affinity = None
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # Look for the first mode line: "   1        -7.5      0.000      0.000"
                # Regex: whitespace, digit, whitespace, float, ...
                match = re.search(r'^\s+1\s+(-?\d+\.\d+)\s+', line)
                if match:
                    best_affinity = float(match.group(1))
                    break
        
        if best_affinity is not None:
            results.append({'ligand': ligand_name, 'affinity': best_affinity})
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('affinity')
        print(df)
        df.to_csv('docking/scores.csv', index=False)
        print("Scores saved to docking/scores.csv")
    else:
        print("No scores found.")

if __name__ == "__main__":
    parse_logs()
