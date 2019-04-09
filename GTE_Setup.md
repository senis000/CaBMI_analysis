
===========

#### Overview
This setup assumes you are running Ubuntu 16: Xenial. The last section will cover steps to run from a fresh installation of Ubuntu.
Ultimately, your directory structure will look like below:
```
CaBMI_Analysis
    |-- te-causality
        |-- transferentropy-sim
            |-- experiments
                |-- example_experiment_1
                    *control.txt
                    *signal.txt
                    |-- outputs
                        *results.mx
        |-- SimKernel-master
        |-- simulation
        |-- gsl
        |-- gsl-1.16
    *utils_gte.py
    *analysis_CaBMI.py
```
Note that the `.gitignore` file will ignore the entire `te-causality` directory.

#### Steps
1. Navigate to the root directory: CaBMI_analysis and clone the GTE repository:
   ```
    git clone https://github.com/olavolav/te-causality
   ```
2. Add the following to your `.bashrc` and source the file:
   ```
    export GTE=/path/to/te-causality
   ```
3. Install the build system and Boost library:
    ```
    sudo apt-get update && \
    sudo apt-get install --yes wget build-essential gcc-multilib libboost-all-dev
   ```
4. Install GSL
   ```
   cd $GTE
   sudo wget -O gsl.tgz http://ftp.gnu.org/gnu/gsl/gsl-1.16.tar.gz \
    && tar -zxf gsl.tgz \
    && mkdir gsl \
    && cd gsl-1.16 \
    && ./configure --prefix=$GTE/gsl \
    && make \
    && sudo make install 
   ```
5. Add the following to your `.bashrc` and source the file:
    ```
    export LIBRARY_PATH=$GTE/gsl/lib/
    export CPLUS_INCLUDE_PATH=$GTE/gsl/include/
    ```
6. Install SimKernel:
    ```
    cd $GTE
    sudo apt-get install --yes unzip \
        && wget -O simkernel.zip http://github.com/ChristophKirst/SimKernel/archive/master.zip \
        && unzip simkernel.zip \
        && cd SimKernel-master \
        && make \
        && sudo make install
    ```
7. Change the following line in your `.bashrc` to the following. Re-source.
    ```
    export LIBRARY_PATH=$GTE/gsl/lib/:$GTE/SimKernel-master/lib/
    ```
8. Install yaml-cpp
    ```
    cd $GTE
    sudo apt-get install --yes cmake \
        && wget -O yaml-cpp.zip https://github.com/jbeder/yaml-cpp/archive/release-0.5.3.zip \
        && unzip yaml-cpp.zip \
        && cd yaml-cpp-release-0.5.3 \
        && mkdir build \
        && cd build \
        && cmake .. \
        && make
    ```
9. Change the previous line in your .bashrc to the following. Re-source.
    ```
    export LIBRARY_PATH=$GTE/gsl/lib/:$GTE/SimKernel-master/lib/:$GTE/yaml-cpp-release-0.5.3/build/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GTE/gsl/lib/:$GTE/SimKernel-master/lib/:$GTE/yaml-cpp-release-0.5.3/build/
    export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$GTE/gsl/include/:$GTE/gsl/lib/:$GTE/yaml-cpp-release-0.5.3/include/
    ```
10. Install Ruby and Rake
    ```
    sudo apt-get install --yes ruby \
        && sudo gem install rake
    ```
11. Build te-causality binaries
    ```
    cd $GTE
    cd transferentropy-sim \
        && rake te-extended test
    ```
11. Execute tests and exit
    ```
    ./test
    ```
12. Make a directory to hold experiment files during GTE execution
    ```
    mkdir $GTE/transferentropy-sim/experiments
    ```

#### Ubuntu Installation
```
sudo add-apt-repository ppa:gnome-terminator
sudo apt-get update
sudo apt-get install terminator
sudo apt-get install python3-pip python-dev build-essential
pip3 install seaborn
pip3 install pandas
pip3 install scikit-image
pip3 install scikit-learn
sudo apt install vim
sudo apt install gnome-tweak-tool
sudo apt-add-repository ppa:numix/ppa
sudo apt-get update
sudo apt-get install numix-icon-theme-circle
```
