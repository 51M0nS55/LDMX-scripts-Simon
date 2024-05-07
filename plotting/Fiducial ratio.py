import numpy as np
import matplotlib.pyplot as plt
import uproot

# Constants for the fiducial cut calculations
scoringPlaneZ = 239.9985
ecalFaceZ = 247.932
cell_radius = 5.0

def projection(Recoilx, Recoily, Recoilz, RPx, RPy, RPz, HitZ):
    """Project the recoil coordinates from the scoring plane to the ecal face."""
    x_final = Recoilx + RPx / RPz * (HitZ - Recoilz) if RPz != 0 else 0
    y_final = Recoily + RPy / RPz * (HitZ - Recoilz) if RPy != 0 else 0
    return (x_final, y_final)

def dist(cell, point):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((cell[0] - point[0])**2 + (cell[1] - point[1])**2)

def load_cellMap(filepath):
    """Load cell map from a given file."""
    cellMap = np.genfromtxt(filepath, dtype={'names': ('x', 'y', 'id'), 'formats': ('f4', 'f4', 'i4')}, usecols=[1,2,0])
    return cellMap

def apply_fiducial_cut(recoilX, recoilY, recoilPx, recoilPy, recoilPz, cells):
    N = len(recoilX)
    f_cut = np.zeros(N, dtype=bool)
    x_final, y_final = projection(recoilX, recoilY, np.full(N, scoringPlaneZ), recoilPx, recoilPy, recoilPz, np.full(N, ecalFaceZ))
    
    for i in range(N):
        if not all([val == -9999 for val in [recoilX[i], recoilY[i], recoilPx[i], recoilPy[i], recoilPz[i]]]):
            for cell in cells:
                if dist(cell, (x_final[i], y_final[i])) <= cell_radius:
                    f_cut[i] = True
                    break
    return f_cut

# Load cell information
cells = load_cellMap('/home/simons/LDMX-scripts-Simon/cellmodule.txt')

# Path to the ROOT file
file_path = '/home/vamitamas/NonFiducialSimu/events_nonfiducial_fullEcal_production.root'

# Open the ROOT file and load the tree
with uproot.open(file_path) as file:
    tree = file["LDMX_Events"]
    # Load the branches for analysis
    recoilX = tree['SimParticles_v14_nonfid.second.x_'].array(library='np')
    recoilY = tree['SimParticles_v14_nonfid.second.y_'].array(library='np')
    recoilPx = tree['SimParticles_v14_nonfid.second.px_'].array(library='np')
    recoilPy = tree['SimParticles_v14_nonfid.second.py_'].array(library='np')
    recoilPz = tree['SimParticles_v14_nonfid.second.pz_'].array(library='np')

    # Apply the fiducial cut
    f_cut = apply_fiducial_cut(recoilX, recoilY, recoilPx, recoilPy, recoilPz, cells)

# Calculate and print statistics
fiducial_events = np.sum(f_cut)
total_events = len(f_cut)
ratio = fiducial_events / total_events
print(f"Total Events: {total_events}")
print(f"Fiducial Events: {fiducial_events}")
print(f"Fiducial/Total Events Ratio: {ratio:.4f}")

# Plotting the histogram of fiducial vs non-fiducial events
plt.hist(f_cut.astype(int), bins=[-0.5, 0.5, 1.5], rwidth=0.8)
plt.xticks([0, 1], ['Non-Fiducial', 'Fiducial'])
plt.xlabel('Event Type')
plt.ylabel('Number of Events')
plt.title('Fiducial vs Non-Fiducial Events')
plt.show()