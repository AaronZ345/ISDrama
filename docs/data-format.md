[Back to ISDrama](../README.md)

# MRSDrama data format

Our dataset is organized hierarchically. 

Each top-level folder contains a set of dramas. Each folder contains a subfolder with cut WAV files, an MP4 video file, and a JSON file containing all annotation information. 
Additionally, the geometric_pose subdirectory stores NumPy (.npy) sequences—listener‑centric 3D positions, head-orientation quaternions, and radial velocities with respect to the left and right ears. These sequences are aligned at the frame level and generated with a 48 kHz sample rate and a 256-sample hop size.
