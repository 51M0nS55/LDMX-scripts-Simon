with uproot.open(filename)['LDMX_Events;5'] as t:
    if not t.keys():  # Ensure the tree has keys
        print(f"FOUND ZOMBIE: {filename}  SKIPPING...", flush=True)
        continue