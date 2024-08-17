import numpy as np
import awkward as ak
import uproot

# File path to analyze
file_path = '/home/vamitamas/Samples8GeV/Ap0.001GeV_sim/*.root'

# Branches we're interested in
branchList = [
    'TargetScoringPlaneHits_signal.x_',
    'TargetScoringPlaneHits_signal.y_',
    'TargetScoringPlaneHits_signal.z_',
    'TargetScoringPlaneHits_signal.energy_'
]

# Parameters for fiducial cuts (e.g., these values should reflect your experiment)
fiducial_cuts = {
    "x_min": -50, "x_max": 50,
    "y_min": -50, "y_max": 50,
    "z_min": 100, "z_max": 200
}

def is_non_fiducial(x, y, z):
    return (x < fiducial_cuts["x_min"]) | (x > fiducial_cuts["x_max"]) | \
           (y < fiducial_cuts["y_min"]) | (y > fiducial_cuts["y_max"]) | \
           (z < fiducial_cuts["z_min"]) | (z > fiducial_cuts["z_max"])

def analyze_file(file_name):
    with uproot.open(file_name) as file:
        tree = file['LDMX_Events;6']
        data = tree.arrays(branchList, interpretation_executor=None)
        
        x = data[branchList[0]]
        y = data[branchList[1]]
        z = data[branchList[2]]
        energy = data[branchList[3]]

        # Find non-fiducial events
        non_fiducial_mask = is_non_fiducial(x, y, z)
        non_fiducial_energy = energy[non_fiducial_mask]
        
        return len(non_fiducial_energy), np.sum(non_fiducial_energy), len(energy)

def compute_non_fiducial_ratio(file_path):
    files = uproot.iterate(f"{file_path}:LDMX_Events;6", expressions=branchList)
    
    total_events = 0
    non_fiducial_events = 0
    non_fiducial_energy_total = 0
    
    for file_name, arrays in files:
        n_non_fiducial, non_fid_energy, n_events = analyze_file(file_name)
        
        total_events += n_events
        non_fiducial_events += n_non_fiducial
        non_fiducial_energy_total += non_fid_energy

    if total_events == 0:
        return None

    ratio = non_fiducial_events / total_events
    uncertainty = np.sqrt(ratio * (1 - ratio) / total_events)

    return {
        "non_fiducial_ratio": ratio,
        "uncertainty": uncertainty,
        "non_fiducial_energy": non_fiducial_energy_total
    }

# Run the analysis
result = compute_non_fiducial_ratio(file_path)
print(result)