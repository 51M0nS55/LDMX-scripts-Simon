import numpy as np

def projection(Recoilx, Recoily, Recoilz, RPx, RPy, RPz, HitZ):
    x_final = Recoilx + RPx/RPz*(HitZ - Recoilz) if RPz != 0 else 0
    y_final = Recoily + RPy/RPz*(HitZ - Recoilz) if RPy != 0 else 0
    return (x_final, y_final)

def dist(cell, point):
    return np.sqrt((cell[0] - point[0])**2 + (cell[1] - point[1])**2)

def load_cellMap(version='v14'):
    cellMap = {}
    for i, x, y in np.loadtxt(f'data/{version}/cellmodule.txt'):
        cellMap[i] = (x, y)
    return np.array(list(cellMap.values()))

def apply_fiducial_cut(recoilX, recoilY, recoilPx, recoilPy, recoilPz, cells):
    N = len(recoilX)
    f_cut = np.zeros(N, dtype=bool)
    for i in range(N):
        if not recoilX[i] == -9999 and not recoilY[i] == -9999 and not recoilPx[i] == -9999 and not recoilPy[i] == -9999 and not recoilPz[i] == -9999:
            fXY = projection(recoilX[i], recoilY[i], scoringPlaneZ, recoilPx[i], recoilPy[i], recoilPz[i], ecalFaceZ)
            for cell in cells:
                if dist(cell, fXY) <= cell_radius:
                    f_cut[i] = True
                    break
    return f_cut

# Constants
scoringPlaneZ = 239.9985
ecalFaceZ = 247.932
cell_radius = 5.0

# Load cell information
cells = load_cellMap()

# Example recoil variables (replace these with actual data)
recoilX, recoilY, recoilPx, recoilPy, recoilPz = np.random.randn(5, 100)  # Dummy data

# Apply fiducial cut
f_cut = apply_fiducial_cut(recoilX, recoilY, recoilPx, recoilPy, recoilPz, cells)

# Now f_cut contains a boolean array indicating whether each event passes the fiducial cut

