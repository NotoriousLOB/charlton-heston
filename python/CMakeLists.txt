# CHARLTON Python Bindings

# Find pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(pybind11)

# Python module
pybind11_add_module(_charlton MODULE
    charlton_python.cpp
)

target_link_libraries(_charlton PRIVATE
    charlton::charlton
)

target_include_directories(_charlton PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Set output location
set_target_properties(_charlton PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
)

# Install
install(TARGETS _charlton
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python
)

# Create Python package structure
set(PYTHON_PACKAGE_DIR ${CMAKE_BINARY_DIR}/python/charlton)
file(MAKE_DIRECTORY ${PYTHON_PACKAGE_DIR})

# Write __init__.py
file(WRITE ${PYTHON_PACKAGE_DIR}/__init__.py ""
"\"\"\"CHARLTON Python Package

Conformal Hyperbolic Accelerated Rough Lévy Transform for Option Numerics

Quick Start:
    import charlton as ch
    
    # Build a model
    model = ch.model().spot(100).rate(0.05).hurst(0.1).from_atm_iv(0.2)
    
    # Create a pricer
    pricer = ch.pricer(model)
    
    # Price options
    put_price = pricer.put(100)
    call_price = pricer.call(100)
    
    # Get Greeks
    greeks = pricer.greeks_put(100, 'cornucopia')
    print(f\"Delta: {greeks['delta']}\")
\"\"\"

from ._charlton import *
from ._charlton import __version__, __author__

# Make model(), pricer(), calibrator() available at package level
model = ModelBuilder
pricer = Pricer
calibrator = Calibrator
"
)

# Copy the built module to the package
add_custom_command(TARGET _charlton POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:_charlton>
        ${PYTHON_PACKAGE_DIR}/_charlton.so
    COMMENT "Copying charlton module to Python package"
)

# Create setup.py for pip installation
file(WRITE ${CMAKE_BINARY_DIR}/python/setup.py ""
"from setuptools import setup, find_packages

setup(
    name='charlton',
    version='1.0.0',
    description='Conformal Hyperbolic Accelerated Rough Lévy Transform for Option Numerics',
    author='CHARLTON Contributors',
    license='MIT',
    packages=find_packages(),
    package_data={
        'charlton': ['_charlton.so'],
    },
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Office/Business :: Financial',
    ],
)
"
)
