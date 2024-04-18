import numpy as np
import matplotlib.pyplot as plt
import uproot
import glob

# Constants for the fiducial cut calculations
scoringPlaneZ = 239.9985
ecalFaceZ = 247.932
cell_radius = 5.0

# Directory names and base directory
base_dir = "/home/vamitamas/Samples8GeV"
batches = [
    "v3.3.3_ecalPN-batch1-trigSkim",
    # ... include all batches ...
]

# Define projection and dist functions here, as per your previous definitions...

def load_cellMap(filepath):
    # Load data assuming the file has three columns: cell_id, x_coord, y_coord
    data = np.loadtxt(filepath, delimiter=',')  # Adjust delimiter as per your file
    cells = {int(row[0]): (row[1], row[2]) for row in data}
    return cells
# Initializing counters
n_initial = 0
n_final = 0
cells = load_cellMap()  # Define load_cellMap or replace with the correct logic to load cells

# Iterate over each directory and process all ROOT files found
for batch in batches:
    root_files = glob.glob(f"{base_dir}/{batch}/*.root")
    for file_path in root_files:
        # Open the file with uproot
        with uproot.open(file_path) as file:
            # Access the TTree
            tree = file['treename']  # Replace 'tree_name' with the actual name of the TTree

            # Extract the branches 
            recoilX = tree['recoilX_branch_name'].array()
            recoilY = tree['recoilY_branch_name'].array()
            recoilZ = tree['recoilZ_branch_name'].array()
            recoilPx = tree['recoilPx_branch_name'].array()
            recoilPy = tree['recoilPy_branch_name'].array()
            recoilPz = tree['recoilPz_branch_name'].array()
            nReadoutHits = tree['nReadoutHits_branch_name'].array()

            # Fiducial cut - the logic to find the e_cut should be defined in a function
            e_cut = find_recoil_electron(recoilX, recoilY, recoilZ, recoilPx, recoilPy, recoilPz)
            f_cut = apply_fiducial_cut(recoilX, recoilY, recoilPx, recoilPy, recoilPz, cells, e_cut)

            # Update the initial and final event counters
            n_initial += len(nReadoutHits)
            n_final += np.sum(f_cut)

# Compute the ratio of the final number of events to the initial number of events
ratio = n_final / n_initial if n_initial > 0 else 0

print(f"Initial number of events: {n_initial}")
print(f"Final number of events after fiducial cut: {n_final}")
print(f"Ratio of final to initial events: {ratio:.4f}")

# Plot histogram of the nReadoutHits after fiducial cut
plt.hist(nReadoutHits[f_cut], bins=50, alpha=0.75, label='After Fiducial Cut')
plt.xlabel('nReadoutHits')
plt.ylabel('Frequency')
plt.legend()
plt.title('Distribution of nReadoutHits After Fiducial Cut')
plt.show()