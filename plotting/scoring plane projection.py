import uproot
import glob

# Function to inspect available fields in one of the ROOT files
def inspect_file(sample_file):
    file_list = glob.glob(sample_file)
    
    if file_list:
        filename = file_list[0]
        print(f"Inspecting file: {filename}")
        
        try:
            with uproot.open(filename) as file:
                print("Available keys in the file:", file.keys())
                
                # Try checking for different cycles
                for cycle in ['7', '6', '5']:  # Check different cycles
                    key = f'LDMX_Events;{cycle}'
                    if key in file.keys():
                        tree = file[key]
                        print(f"Available branches in '{key}':", tree.keys())
                        break
                else:
                    print("No suitable 'LDMX_Events' tree found in this file.")
        
        except Exception as e:
            print(f"Error while inspecting file {filename}: {e}")
    else:
        print(f"No files found at {sample_file}")

# Sample file to inspect
sample_file = '/home/vamitamas/Samples8GeV/Ap0.001GeV_sim/*.root'

# Run inspection on the sample file
inspect_file(sample_file)