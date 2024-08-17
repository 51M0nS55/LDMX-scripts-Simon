import numpy as np
import uproot
import glob
import re
import math
from tqdm import tqdm
import awkward
import concurrent.futures
import json

executor = concurrent.futures.ThreadPoolExecutor(20)

# detector constants
SP_TARGET_DOWN_Z = 0.1767
ECAL_SP_Z = 239.9985
ECAL_FACE_Z = 247.932
CELL_RADIUS = 5.0

# Functions for distance and projection calculations
def dist(p1, p2):
    return math.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))

def projection(Recoilx, Recoily, Recoilz, RPx, RPy, RPz, HitZ):
    x_final = Recoilx + RPx / RPz * (HitZ - Recoilz) if RPz else 0
    y_final = Recoily + RPy / RPz * (HitZ - Recoilz) if RPy else 0
    return (x_final, y_final)

def _load_cellMap(version='v13'):
    cellMap = {}
    for i, x, y in np.loadtxt(f'/home/duncansw/LDMX-scripts/GraphNet/data/{version}/cellmodule.txt'):
        cellMap[i] = (x, y)
    global cells
    cells = np.array(list(cellMap.values()))
    print("Loaded {} detector info".format(version))

def pad_array(arr):
    arr = awkward.pad_none(arr, 1, clip=True)
    arr = awkward.fill_none(arr, 0)
    return awkward.flatten(arr)

# v14 8gev files
file_templates = {
    0.001: '/home/vamitamas/Samples8GeV/Ap0.001GeV_sim/*.root',
    0.01: '/home/vamitamas/Samples8GeV/Ap0.01GeV_sim/*.root',
    0.1: '/home/vamitamas/Samples8GeV/Ap0.1GeV_sim/*.root',
    1.0: '/home/vamitamas/Samples8GeV/Ap1GeV_sim/*.root',
    0: '/home/vamitamas/Samples8GeV/v3.3.3_ecalPN*/*.root'
}

# Load ecal cell geometry
_load_cellMap()

# Dictionary for nonfiducial ratios (for each mass point)
nonfid_ratios = {}
# Loop over mass points
for mass in file_templates.keys():

    print(f"==== m = {mass} ====", flush=True)

    # Branch lists for signal vs. background
    if mass:
        branchList = ['TargetScoringPlaneHits_signal/TargetScoringPlaneHits_signal.pdgID_', 'TargetScoringPlaneHits_signal/TargetScoringPlaneHits_signal.x_',
                      'TargetScoringPlaneHits_signal/TargetScoringPlaneHits_signal.y_', 'TargetScoringPlaneHits_signal/TargetScoringPlaneHits_signal.z_',
                      'TargetScoringPlaneHits_signal/TargetScoringPlaneHits_signal.px_', 'TargetScoringPlaneHits_signal/TargetScoringPlaneHits_signal.py_',
                      'TargetScoringPlaneHits_signal/TargetScoringPlaneHits_signal.pz_', 'TriggerSums20Layers_signal/pass_',
                      'EcalRecHits_signal/EcalRecHits_signal.energy_', 'EcalRecHits_signal/EcalRecHits_signal.isNoise_']
    else:
        branchList = ['TargetScoringPlaneHits_sim/TargetScoringPlaneHits_sim.pdgID_', 'TargetScoringPlaneHits_sim/TargetScoringPlaneHits_sim.x_',
                      'TargetScoringPlaneHits_sim/TargetScoringPlaneHits_sim.y_', 'TargetScoringPlaneHits_sim/TargetScoringPlaneHits_sim.z_',
                      'TargetScoringPlaneHits_sim/TargetScoringPlaneHits_sim.px_', 'TargetScoringPlaneHits_sim/TargetScoringPlaneHits_sim.py_',
                      'TargetScoringPlaneHits_sim/TargetScoringPlaneHits_sim.pz_', 'EcalRecHits_sim/EcalRecHits_sim.energy_',
                      'EcalRecHits_sim/EcalRecHits_sim.isNoise_']

    file_list = glob.glob(file_templates[mass])
    nFiles = len(file_list)

    nEvents = 0  # Count total events (post-trigger)
    nNonFid = 0  # Count nonfiducial events
    ecal_energy_hits = []  # Store ECal energy hits for non-fiducial events

    # Loop over files for this mass
    for i, filename in tqdm(enumerate(file_list), total=nFiles):
        if nEvents >= 100:
            break
        try:
            with uproot.open(filename, interpretation_executor=executor)['LDMX_Events;6'] as t:
                if not t.keys():
                    print(f"FOUND ZOMBIE: {filename}  SKIPPING...", flush=True)
                    continue
                key_miss = False
                for branch in branchList:
                    if not re.split('/', branch)[0] in t.keys():
                        key_miss = True
                        break
                if key_miss:
                    print(f"MISSING KEY(S) IN: {filename}  SKIPPING...", flush=True)
                    continue

                data = t.arrays(branchList, interpretation_executor=executor)

                # tsp leaves
                pdgID = data[branchList[0]]
                x = data[branchList[1]]
                y = data[branchList[2]]
                z = data[branchList[3]]
                px = data[branchList[4]]
                py = data[branchList[5]]
                pz = data[branchList[6]]

                # Apply trigger for signal
                if mass:
                    tskimmed_data = {}
                    trig_pass = data[branchList[7]]
                    for branch in branchList:
                        tskimmed_data[branch] = data[branch][trig_pass]
                    pdgID = tskimmed_data[branchList[0]]
                    x = tskimmed_data[branchList[1]]
                    y = tskimmed_data[branchList[2]]
                    z = tskimmed_data[branchList[3]]
                    px = tskimmed_data[branchList[4]]
                    py = tskimmed_data[branchList[5]]
                    pz = tskimmed_data[branchList[6]]

                # Add events to running count
                nEvents += len(z)

                # Select recoil electron at downstream tsp (maximal forward moving pz electron)
                e_cut = np.zeros_like(px, dtype=bool).tolist()
                for i in range(len(px)):
                    maxPz = 0
                    e_index = -1
                    for j in range(len(px[i])):
                        if pdgID[i][j] == 11 and z[i][j] > 0.17 and z[i][j] < 0.18 and pz[i][j] > maxPz:
                            maxPz = pz[i][j]
                            e_index = j
                    if e_index >= 0:
                        e_cut[i][e_index] = True

                e_cut = awkward.Array(e_cut)
                recoilX = pad_array(x[e_cut])
                recoilY = pad_array(y[e_cut])
                recoilPx = pad_array(px[e_cut])
                recoilPy = pad_array(py[e_cut])
                recoilPz = pad_array(pz[e_cut])

                # Fiducial cut
                N = len(recoilX)
                f_cut = np.zeros(N, dtype=bool)
                for i in range(N):
                    fiducial = False
                    fXY = projection(recoilX[i], recoilY[i], SP_TARGET_DOWN_Z, recoilPx[i], recoilPy[i], recoilPz[i], ECAL_FACE_Z)
                    if not (recoilX[i] == 0 and recoilY[i] == 0 and recoilPx[i] == 0 and recoilPy[i] == 0 and recoilPz[i] == 0):
                        for j in range(len(cells)):
                            celldis = dist(cells[j], fXY)
                            if celldis <= CELL_RADIUS:
                                fiducial = True
                                break
                    if fiducial:
                        f_cut[i] = True

                # Add nonfiducial count to running total
                non_fid_events = f_cut == 0
                nNonFid += np.sum(non_fid_events)

                # Extract ECal energy for non-fiducial events
                energy = data[branchList[8]]
                isNoise = data[branchList[9]]

                for event in range(len(energy)):
                    for hit in range(len(energy[event])):
                        if not isNoise[event][hit] and non_fid_events[event]:
                            ecal_energy_hits.append(energy[event][hit])

        except OSError:
            continue

    # Compute nonfiducial ratio (for this mass point)
    if nEvents > 0:
        nonfid_ratio = nNonFid / nEvents
        nonfid_uncertainty = nonfid_ratio * (math.sqrt(nNonFid) / nNonFid + math.sqrt(nEvents) / nEvents)
        nonfid_ratios[mass] = {
            "ratio": nonfid_ratio,
            "uncertainty": nonfid_uncertainty,
            "ecal_energy_hits": ecal_energy_hits
        }
    else:
        nonfid_ratios[mass] = "no events"

# Extract ECal energy for non-fiducial events
if not mass:  # only for background
    energy = data['EcalRecHits_signal/EcalRecHits_signal.energy_']
    isNoise = data['EcalRecHits_signal/EcalRecHits_signal.isNoise_']
    
    # Loop over events and energy hits
    for event in range(len(energy)):
        if event >= len(non_fid_events):
            continue  # Skip if the event index is out of range for non_fid_events
        for hit in range(len(energy[event])):
            if hit >= len(isNoise[event]):
                continue  # Skip if the hit index is out of range for isNoise
            if not isNoise[event][hit] and non_fid_events[event]:
                hit_energy = energy[event][hit]
                ecal_energy_hits.append(hit_energy)


# Print non-fiducial ratios with uncertainties
print(json.dumps(nonfid_ratios, indent=4))