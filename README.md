# QuadrupedWholeBodyMPC (WIP)

## Conventions
We are currently switching to a new nominal convention for the state vector of the robot which will be used for all code within this repo. The following
subsets of code have not been updated to the new convention yet:
- src/dynamics examples/tests
- src/altro-c
- src/hybrid-trajopt-ipopt and its corresponding examples/tests
- src/hybrid-trajopt-altro and its corresponding examples/tests
- src/ros examples/tests

The following subsets have been updated:
- src/dynamics
- src/ros
- hybrid_trajopt_altro references

### Nominal (our) conventions:
   Config vector:
   
       [x, y, z, quat_w, quat_x, quat_y, quat_z, FL_hip, FL_thigh, FL_calf,
        FR_hip, FR_thigh, FR_calf, RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]
   Velocity vector (body velocities in body frame):
   
       [v_x, v_y, v_z, ω_x, ω_y, ω_z, FL_hip, FL_thigh, FL_calf,
        FR_hip, FR_thigh, FR_calf, RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]
        
### Pinocchio conventions (only differs in config):
   Config vector:
   
       [x, y, z, quat_x, quat_y, quat_z, quat_w, FL_hip, FL_thigh, FL_calf,
        FR_hip, FR_thigh, FR_calf, RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]
        
   Velocity vector (body velocities in body frame):
   
       [v_x, v_y, v_z, ω_x, ω_y, ω_z, FL_hip, FL_thigh, FL_calf,
        FR_hip, FR_thigh, FR_calf, RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]

### RigidBodyDynamics conventions:
   Config vector:
   
       [quat_w, quat_x, quat_y, quat_z, x, y, z, FR_hip, FL_hip, RR_hip, RL_hip,
        FR_thigh, FL_thigh, RR_thigh, RL_thigh, FR_calf, FL_calf, RR_calf, RL_calf]
   Velocity vector (body velocities in body frame):
   
       [ω_x, ω_y, ω_z, v_x, v_y, v_z, FR_hip, FL_hip, RR_hip, RL_hip,
        FR_thigh, FL_thigh, RR_thigh, RL_thigh, FR_calf, FL_calf, RR_calf, RL_calf]
        
### Switching between conventions (WIP)

src/dynamics/conversions.jl provides the following helper functions to convert between the different conventions. It includes the following functions: change_order, change_order!, change_orders, and change_orders!. They can be used in the following way:

```
# For vectors
test = rand(19) # Random config
test2 = change_order(test, :nominal, :pinocchio)
change_order!(test2, :pinocchio, :nominal) # Should now match test 1

# Also works with matrices (need to specify dimensions, default is (1, 2))
test = rand(19, 3)
test2 = change_order(test, :nominal, :pinocchio, dims = (1,))
test3 = change_order(test', :nominal, :pinocchio, dims = (2,))

# And works with multiple inputs
config = rand(19)
vel = rand(18)
change_orders!([config, vel], :nominal, :pinocchio)

# For multiple matrices, can specify dims
J = rand(19, 12) # Want to change rows
J_d = rand(12, 18) # Want to change cols
change_orders!([J, J_d], :nominal, :pinocchio, [(1,), (2,)]
```

The code assumes that 18-dim vectors are velocities/error config vectors, 19-dim vectors are configurations, 36-dim vectors are error states, and 37-dim vectors are states. When working with matrices, you can use dims to leave either the rows or columns alone if desired, even if the sizing matches one of the four above. 

## Setup
The following installation steps have only been tested on Ubuntu 20.04 LTS. You'll need to have CMake and Julia (we used v1.8) installed.

First clone the repository, cd into the main folder and run the following to clone the Altro submodule.
```
git submodule update --init --recursive
```


### Install Pinocchio
Install dependencies
```
sudo apt install -qqy lsb-release gnupg2 curl
```
Add robotpkg as source repository to apt
```
echo "deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" | sudo tee /etc/apt/sources.list.d/robotpkg.list
```
Register the authentication certificate of robotpkg
```
curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -
```
Run apt-get update to fetch the package descriptions
```
sudo apt-get update
```
Install Pinocchio
```
sudo apt install -qqy robotpkg-py38-pinocchio 
```
If you're using Jammy, you can try robotpgk-py310-pinocchio.

Finally, setup environment variables (copy the following lines into ~/.bashrc to have the env variables persist between sessions).
```
export PATH=/opt/openrobots/bin:$PATH
export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/openrobots/lib/python3.8/site-packages:$PYTHONPATH # Adapt your desired python version here
export CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH
```
These instructions were copied from [Pinocchio's Documentation](https://stack-of-tasks.github.io/pinocchio/download.html) 
on February 8th, 2023 for a specific Python version (3.8). Installation instructions may have changed since then.

### Install CasADI
These instructions were copied from [CasADI's Documentation](https://github.com/casadi/casadi/wiki/InstallationLinux) on June 18th, 2023.
```
sudo apt-get install gcc g++ gfortran git cmake liblapack-dev pkg-config --install-recommends -y
sudo apt-get install coinor-libipopt-dev -y # Install IPOPT (optional)
git clone https://github.com/casadi/casadi.git -b main casadi
cd casadi && git checkout 3.6.3
mkdir build && cd build
cmake ..
make
sudo make install
```

### Install MuJoCo.jl
In Julia, activate the QuadrupedWholeBodyMPC environment and then run the following
```
dev submodules/MuJoCo.jl
```

### Generate the C shared dynamics library for the go1
Once you've cloned the repo, do the following from the QuadrupedWholeBodyMPC folder.
```
mkdir build
cd build
cmake .. 
cmake --build .
```
This should create a file in the build directory named libgo1_dynamics.so, which will be loaded by src/go1_dynamics.jl. To test that the shared library file
was created correctly, run the following from the build directory.
```
julia ../test/dynamics/test_go1_dynamics.jl
```
This script should instantiate the julia environment and install the necessary packages (if not already done), and then print the following output:
```
Loading robot from URDF
Configuration dimension: 19
Velocity dimension: 18
Testing dynamics functions are self-consistent
Test Passed
Rank of kinematics jacobian in default config: 12
```
