
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

### Generate the C shared dynamics library for the go1
Once you've cloned the repo, do the following from the QuadrupedWholeBodyMPC folder.
```
mkdir build
cd build
cmake .. 
cmake --build .
```
This should create a file in the build directory named libgo1_dynamics.so, which will be loaded by src/go1_dynamics.jl. T
