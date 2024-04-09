
echo off
echo Deleting cpp and pyd files in Pygameshader directory prior installation
cd Cielab
del *.cpp
del *.pyd
cd ..

echo Building Cielab with python ver %python3_6%
call %python3_6% setup.py build_ext --inplace
call %python3_6% setup.py bdist_wheel
echo Cleaning up last build
cd Cielab
del *.cpp
del *.pyd
cd ..

echo Building Cielab with python ver %python3_7%
call %python3_7% setup.py build_ext --inplace
call %python3_7% setup.py bdist_wheel
echo Cleaning up last build
cd Cielab
del *.cpp
del *.pyd
cd ..
echo Building Cielab with python ver %python3_8%
call %python3_8% setup.py build_ext --inplace
call %python3_8% setup.py bdist_wheel
echo Cleaning up last build
cd Cielab
del *.cpp
del *.pyd
cd ..
echo Building Cielab with python ver %python3_9%
call %python3_9% setup.py build_ext --inplace
call %python3_9% setup.py bdist_wheel
echo Cleaning up last build
cd Cielab
del *.cpp
del *.pyd
cd ..
echo Building Cielab with python ver %python3_10%
call %python3_10% setup.py build_ext --inplace
call %python3_10% setup.py bdist_wheel
echo Cleaning up last build
cd Cielab
del *.cpp
del *.pyd
cd ..
echo Building Cielab with python ver %python3_11%
call %python3_11% setup.py build_ext --inplace
call %python3_11% setup.py sdist bdist_wheel
echo Cleaning up last build
cd Cielab
del *.cpp
del *.pyd
cd ..

echo Building Cielab with python ver %python3_12%
call %python3_12% setup.py build_ext --inplace
call %python3_12% setup.py sdist bdist_wheel
echo Cleaning up last build
cd Cielab
del *.cpp
del *.pyd
cd ..
