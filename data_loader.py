from pathlib import Path
import pandas as pd


def load_shots(data_dir: str = "data") -> dict:
    """Load all shot files and map to id."""
    data_path = Path(data_dir)

    # Load all shot files
    files = sorted(data_path.glob("*.txt"))
    
    shots = {}
    for idx, file in enumerate(files, start=1):
        time_series = {}
        metadata = {}
        
        with open(file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # parse "metric_name {val1 val2 ...}" format
                if '{' in line and '}' in line:
                    metric_name = line.split('{')[0].strip()
                    values_str = line.split('{')[1].split('}')[0].strip()
                    vals = [float(v) for v in values_str.split()]
                    time_series[metric_name] = vals
                # parse "metric_name value" format
                elif ' ' in line:
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        metric_name, value = parts
                        try:
                            metadata[metric_name] = float(value)
                        except ValueError:
                            metadata[metric_name] = value
        
        if time_series:
            # Handle unequal array lengths
            max_len = max(len(v) for v in time_series.values())
            for key in time_series:
                if len(time_series[key]) < max_len:
                    # pad with last value
                    time_series[key] = time_series[key] + [time_series[key][-1]] * (max_len - len(time_series[key]))

            df = pd.DataFrame(time_series)
            # Store metadata as attrs
            df.attrs['metadata'] = metadata
            shots[idx] = df
    
    return shots


if __name__ == "__main__":
    shots = load_shots()
    print(f"Loaded {len(shots)} shots")
    if shots:
        print(f"First shot columns: {list(shots[1].columns)}")
    
    # self test
    assert len(shots) == 13, f"Expected 13 shots, got {len(shots)}"
    print("Self-test passed!")

