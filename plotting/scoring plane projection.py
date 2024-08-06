import numpy as np
import uproot
import glob
import re
import math
from tqdm import tqdm
import awkward as ak
import concurrent.futures
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
SP_TARGET_DOWN_Z = 0.1767
ECAL_SP_Z = 239.9985
ECAL_FACE_Z = 247.932
CELL_RADIUS = 5.0

def dist(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))

def projection(Recoilx, Recoily, Recoilz, RPx, RPy, RPz, HitZ):
    x_final = Recoilx + RPx / RPz * (HitZ - Recoilz) if RPz else 0
    y_final = Recoily + RPy / RPz * (HitZ - Recoilz) if RPy else 0
    return (x_final, y_final)

def load_cellMap(version='v13'):
    cellMap = {}
    try:
        for i, x, y in np.loadtxt(f'/home/duncansw/LDMX-scripts/GraphNet/data/{version}/cellmodule.txt'):
            cellMap[i] = (x, y)
    except IOError as e:
        logging.error(f"Failed to load cell map: {e}")
        raise
    cells = np.array(list(cellMap.values()))
    logging.info(f"Loaded {version} detector info with {len(cells)} cells")
    return cells

def pad_array(arr):
    arr = ak.pad_none(arr, 1, clip=True)
    arr = ak.fill_none(arr, 0)
    return ak.flatten(arr)

def apply_fiducial_cut(recoilX, recoilY, recoilPx, recoilPy, recoilPz, cells):
    N = len(recoilX)
    f_cut = np.zeros(N, dtype=bool)
    fXY = projection(recoilX, recoilY, SP_TARGET_DOWN_Z, recoilPx, recoilPy, recoilPz, ECAL_FACE_Z)

    for i in range(N):
        if not (recoilX[i] == 0 and recoilY[i] == 0 and recoilPx[i] == 0 and recoilPy[i] == 0 and recoilPz[i] == 0):
            for j in range(len(cells)):
                if dist(cells[j], fXY[i]) <= CELL_RADIUS:
                    f_cut[i] = True
                    break
    return f_cut

def process_file(filename, branchList, cells, mass):
    nEvents = 0
    nNonFid = 0
    ecal_energies = []

    try:
        with uproot.open(filename) as file:
            if not file.keys():
                logging.warning(f"FOUND ZOMBIE: {filename}  SKIPPING...")
                return nEvents, nNonFid, ecal_energies

            with file['LDMX_Events'] as t:
                if not t.keys():
                    logging.warning(f"FOUND ZOMBIE: {filename}  SKIPPING...")
                    return nEvents, nNonFid, ecal_energies

                key_miss = any(re.split('/', branch)[0] not in t.keys() for branch in branchList)
                if key_miss:
                    logging.warning(f"MISSING KEY(S) IN: {filename}  SKIPPING...")
                    return nEvents, nNonFid, ecal_energies

                data = t.arrays(branchList)

                # TSP leaves
                pdgID = data[branchList[0]]
                x = data[branchList[1]]
                y = data[branchList[2]]
                z = data[branchList[3]]
                px = data[branchList[4]]
                py = data[branchList[5]]
                pz = data[branchList[6]]

                # Apply trigger for signal
                if mass:
                    trig_pass = data[branchList[7]]
                    pdgID = pdgID[trig_pass]
                    x = x[trig_pass]
                    y = y[trig_pass]
                    z = z[trig_pass]
                    px = px[trig_pass]
                    py = py[trig_pass]
                    pz = pz[trig_pass]

                # Add events to running count
                nEvents += len(z)

                # Select recoil electron at downstream TSP (maximal forward moving pz electron)
                e_cut = np.zeros_like(px, dtype=bool)
                for i in range(len(px)):
                    maxPz = 0
                    e_index = -1
                    for j in range(len(px[i])):
                        if pdgID[i][j] == 11 and z[i][j] > 0.17 and z[i][j] < 0.18 and pz[i][j] > maxPz:
                            maxPz = pz[i][j]
                            e_index = j
                    if e_index >= 0:
                        e_cut[i][e_index] = True

                recoilX = pad_array(x[e_cut])
                recoilY = pad_array(y[e_cut])
                recoilPx = pad_array(px[e_cut])
                recoilPy = pad_array(py[e_cut])
                recoilPz = pad_array(pz[e_cut])

                # Fiducial cut
                f_cut = apply_fiducial_cut(recoilX, recoilY, recoilPx, recoilPy, recoilPz, cells)

                # Add non-fiducial count to running total
                non_fid_events = recoilX[f_cut == 0]
                nNonFid += len(non_fid_events)

                # Extract ECal energy for non-fiducial events
                for event in non_fid_events:
                    ecal_energy = data['ECalHits_sim.energy_'][event]
                    ecal_energies.extend(ecal_energy)

    except Exception as e:
        logging.error(f"Error processing file {filename}: {e}")

    return nEvents, nNonFid, ecal_energies

def main():
    # Load ecal cell geometry
    cells = load_cellMap()

    # v14 8gev files
    file_templates = {
        0.001:  '/home/vamitamas/Samples8GeV/Ap0.001GeV_sim/*.root',
        0.01:  '/home/vamitamas/Samples8GeV/Ap0.01GeV_sim/*.root',
        0.1:   '/home/vamitamas/Samples8GeV/Ap0.1GeV_sim/*.root',
        1.0:   '/home/vamitamas/Samples8GeV/Ap1GeV_sim/*.root',
        0:     '/home/vamitamas/Samples8GeV/v3.3.3_ecalPN*/*.root'
    }

    nonfid_ratios = {}
    for mass, pattern in file_templates.items():
        logging.info(f"==== Processing mass {mass} ====")
        branchList = [
            'TargetScoringPlaneHits_signal/TargetScoringPlaneHits_signal.pdgID_', 
            'TargetScoringPlaneHits_signal/TargetScoringPlaneHits_signal.x_',
            'TargetScoringPlaneHits_signal/TargetScoringPlaneHits_signal.y_', 
            'TargetScoringPlaneHits_signal/TargetScoringPlaneHits_signal.z_',
            'TargetScoringPlaneHits_signal/TargetScoringPlaneHits_signal.px_', 
            'TargetScoringPlaneHits_signal/TargetScoringPlaneHits_signal.py_', 
            'TargetScoringPlaneHits_signal/TargetScoringPlaneHits_signal.pz_', 
            'TriggerSums20Layers_signal/pass_'
        ] if mass else [
            'TargetScoringPlaneHits_sim/TargetScoringPlaneHits_sim.pdgID_', 
            'TargetScoringPlaneHits_sim/TargetScoringPlaneHits_sim.x_',
            'TargetScoringPlaneHits_sim/TargetScoringPlaneHits_sim.y_', 
            'TargetScoringPlaneHits_sim/TargetScoringPlaneHits_sim.z_',
            'TargetScoringPlaneHits_sim/TargetScoringPlaneHits_sim.px_', 
            'TargetScoringPlaneHits_sim/TargetScoringPlaneHits_sim.py_', 
            'TargetScoringPlaneHits_sim/TargetScoringPlaneHits_sim.pz_',
            'ECalHits_sim.energy_'
        ]

        file_list = glob.glob(pattern)
        nFiles = len(file_list)

        total_events = 0
        total_nonfid = 0
        all_ecal_energies = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            results = list(tqdm(executor.map(lambda f: process_file(f, branchList, cells, mass), file_list), total=nFiles))

        for nEvents, nNonFid, ecal_energies in results:
            total_events += nEvents
            total_nonfid += nNonFid
            all_ecal_energies.extend(ecal_energies)

        if total_events > 0:
            nonfid_ratio = total_nonfid / total_events
            nonfid_uncertainty = nonfid_ratio * (np.sqrt(total_nonfid) / total_nonfid + np.sqrt(total_events) / total_events)
            nonfid_ratios[mass] = {
                "ratio": nonfid_ratio,
                "uncertainty": nonfid_uncertainty
            }
        else:
            nonfid_ratios[mass] = "no events"

        # Further analysis on ECal energies
        # You can add more detailed analysis here as per your requirements
        logging.info(f"ECal energies for mass {mass}: {all_ecal_energies}")

    # Print non-fiducial ratios with uncertainties
    import json
    print(json.dumps(nonfid_ratios, indent=4))

if __name__ == "__main__":
    main()