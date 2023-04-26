
## License

Please request a license for dco/c++ from christodoulou(at)stce.rwth-aachen.de to run this code. 

## Integrating custom Smoothed Functions via pybind

Custom functions can be defined in header-file source/pybind_integrated/smoothing_examples.hpp, and have to be exposed to python in source/pybind_integrated/pybinder.cpp via pybind. They are then compiled into a python module. 



## Building

Building is done in CMake. CMake also downloads dco 3 (if not present) if it isn't present or if it can't be detected. 
Do the build from the source-folder.

```
cd build 
cmake ../source && make
cd -

```



## Running the examples

Inside the project-folder, there's plot_example.py, which uses bindings from C++ into python (bound with pybind). 
```
python3 plot_example.py
```

To recreate figures and data used in the paper, navigate to
```
cd tfops
```
and run the tensorflow-bindings, where gradient-descent is performed.


## Defining more functions

In /source/pybind_integrated there are the files pybinder.cpp and smoothing_examples.hpp. We can edit them for integrating new functions.

In the python-file in the project-directory, you can import the newly bound functions from the module as follows: 
```python
from source.build.pybind_integrated.Smoothing import set_smfactor, cb2
```
(to import functions 'set_smfactor' and 'cb2')

