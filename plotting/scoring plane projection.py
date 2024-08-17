import uproot

file_path = "/home/vamitamas/Samples8GeV/Ap0.001GeV_sim/mc_v14-8gev-8.0GeV-1e-signal_W_noDecay_sim_run10025_t1699047029.root"

with uproot.open(file_path) as file:
    print("Keys:", file.keys())
    if 'LDMX_Events' in file.keys():
        tree = file['LDMX_Events']
        print("Branches in 'LDMX_Events':", tree.keys())
    else:
        print("'LDMX_Events' not found in this file.")