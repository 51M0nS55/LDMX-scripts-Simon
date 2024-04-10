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
directories = [
    "v3.3.3_ecalPN-batch1-trigSkim",
    "v3.3.3_ecalPN-batch2-trigSkim",
    "v3.3.3_ecalPN-batch3-trigSkim",
    "v3.3.3_ecalPN-batch4-trigSkim",
    "v3.3.3_ecalPN-batch5-trigSkim",
    "v3.3.3_ecalPN-batch6-trigSkim",
    "v3.3.3_ecalPN-batch7-trigSkim",
    "v3.3.3_ecalPN-batch8-trigSkim",
]

# Function to calculate the projection to the ecal face
def projection(Recoilx, Recoily, Recoilz, RPx, RPy, RPz, HitZ):
    x_final = Recoilx + RPx/RPz * (HitZ - Recoilz) if RPz != 0 else 0
    y_final = Recoily + RPy/RPz * (HitZ - Recoilz) if RPz != 0 else 0
    return (x_final, y_final)

# Function to calculate distance
def dist(cell, point):
    return np.sqrt((cell[0] - point[0])**2 + (cell[1] - point[1])**2)

# Function to load the cell map from a given version
def load_cellMap(version='v14'):
    cellMap = {}
    # Adjust the path to the cellmodule.txt file as needed
    for i, x, y in np.loadtxt(f'data/{version}/cellmodule.txt', unpack=True):
        cellMap[i] = (x, y)
    return np.array(list(cellMap.values()))

# Function to apply the fiducial cut
def apply_fiducial_cut(recoilX, recoilY, recoilPx, recoilPy, recoilPz, cells):
    N = len(recoilX)
    f_cut = np.zeros(N, dtype=bool)
    for i in range(N):
        fiducial = False
        fXY = projection(recoilX[i], recoilY[i], scoringPlaneZ, recoilPx[i], recoilPy[i], recoilPz[i], ecalFaceZ)
        if not all(val == -9999 for val in [recoilX[i], recoilY[i], recoilPx[i], recoilPy[i], recoilPz[i]]):
            for cell in cells:
                if dist(cell, fXY) <= cell_radius:
                    fiducial = True
                    break
        f_cut[i] = fiducial
    return f_cut

# Load cell information
cells = load_cellMap()

# Initialize a list to hold the fiducial cut flags for all processed events
all_f_cut = []

# Iterate over each directory and process all ROOT files found
for directory in directories:
    root_files = glob.glob(f"{base_dir}/{directory}/*.root")
    for file_path in root_files:
        # Here, you would load your data using uproot
        # For example: with uproot.open(file_path) as file: ...
        # Dummy data for illustration purposes:
        N = 100  # Assuming 100 events per file; replace with dynamic loading
        recoilX, recoilY, recoilPx, recoilPy, recoilPz = np.random.rand(5, N)  # Dummy data
        
        # Apply fiducial cut to loaded data
        f_cut = apply_fiducial_cut(recoilX, recoilY, recoilPx, recoilPy, recoilPz, cells)
        all_f_cut.extend(f_cut)

# Convert list of flags to a NumPy array for easier analysis
all_f_cut = np.array(all_f_cut)

# Calculate and print statistics
fiducial_events = np.sum(all_f_cut)
total_events = len(all_f_cut)
ratio = fiducial_events / total_events
print(f"Fiducial/Total Events Ratio: {ratio:.4f}")

# Plotting the histogram of fiducial vs non-fiducial events
plt.hist(all_f_cut.astype(int), bins=[-0.5, 0.5, 1.5], rwidth=0.8, labels=['Non-Fiducial', 'Fiducial'])
plt.xticks([0, 1])
plt.xlabel('Event Type')
plt.ylabel('Number of Events')
plt.title('Fiducial vs Non-Fiducial Events')
plt.show()
