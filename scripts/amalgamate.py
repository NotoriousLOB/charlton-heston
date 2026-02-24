#!/usr/bin/env python3
"""
CHARLTON Amalgamation Script

Generates a single-header version of CHARLTON by combining all modular headers.
Similar to SQLite's amalgamation process.

Usage:
    python3 scripts/amalgamate.py
    
Output:
    include/charlton.hpp - Single amalgamated header
"""

import os
import re
from pathlib import Path

# Header order matters for dependencies
HEADER_ORDER = [
    "core.hpp",
    "simd.hpp", 
    "types.hpp",
    "aad.hpp",
    "csd.hpp",
    "abm_solver.hpp",
    "pricer.hpp",
    "greeks.hpp",
    "calibrator.hpp",
]

HEADER_DIR = Path("include/charlton")
OUTPUT_FILE = Path("include/charlton.hpp")

def extract_content(filepath):
    """Extract the content of a header file, removing guards, namespace, and includes."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Remove file-level doc comment
    content = re.sub(r'^/\*\*.*?\*/\s*\n', '', content, count=1, flags=re.DOTALL)
    
    # Remove include guards
    content = re.sub(r'#ifndef\s+\w+_HPP\s*\n#define\s+\w+_HPP\s*\n', '', content)
    content = re.sub(r'\n#endif\s*//\s*\w+_HPP\s*\n?$', '\n', content)
    
    # Remove includes of other charlton headers (both quoted and bracketed)
    content = re.sub(r'#include\s+["<]charlton/[^">]+[">]\s*\n', '', content)
    content = re.sub(r'#include\s+"core\.hpp"\s*\n', '', content)
    content = re.sub(r'#include\s+"simd\.hpp"\s*\n', '', content)
    content = re.sub(r'#include\s+"types\.hpp"\s*\n', '', content)
    content = re.sub(r'#include\s+"aad\.hpp"\s*\n', '', content)
    content = re.sub(r'#include\s+"csd\.hpp"\s*\n', '', content)
    content = re.sub(r'#include\s+"abm_solver\.hpp"\s*\n', '', content)
    content = re.sub(r'#include\s+"pricer\.hpp"\s*\n', '', content)
    content = re.sub(r'#include\s+"greeks\.hpp"\s*\n', '', content)
    content = re.sub(r'#include\s+"calibrator\.hpp"\s*\n', '', content)
    
    # Remove namespace declaration and closing brace
    content = re.sub(r'namespace\s+charlton\s*\{', '', content, count=1)
    content = re.sub(r'\}\s*//\s*namespace\s+charlton', '', content)
    
    # Remove empty #ifdef blocks that might be left over
    content = re.sub(r'#ifdef [A-Z_]+\s*\n#endif\s*\n', '', content)
    
    return content.strip()

def generate_amalgamation():
    """Generate the amalgamated header file."""
    
    lines = []
    
    # File header
    lines.append('/**')
    lines.append(' * CHARLTON - Conformal Hyperbolic Accelerated Rough L\u00e9vy Transform for Option Numerics')
    lines.append(' * ')
    lines.append(' * A high-performance C++ library for pricing and calibration in the Rough Heston model.')
    lines.append(' * ')
    lines.append(' * This is the AMALGAMATED header file - all modules combined into a single header.')
    lines.append(' * For development, see the modular headers in include/charlton/')
    lines.append(' * ')
    lines.append(' * To regenerate this file from the modular headers:')
    lines.append(' *   python3 scripts/amalgamate.py')
    lines.append(' * ')
    lines.append(' * Backronym: "You can have my Bermudan swaptions when you pry them from my')
    lines.append(' *            COLD DEAD HAND!!!" - Charlton Heston')
    lines.append(' * ')
    lines.append(' * Copyright (c) 2025 - MIT License')
    lines.append(' */')
    lines.append('')
    lines.append('#ifndef CHARLTON_HPP')
    lines.append('#define CHARLTON_HPP')
    lines.append('')
    lines.append('// ============================================================================')
    lines.append('// Standard Library Includes')
    lines.append('// ============================================================================')
    lines.append('')
    lines.append('#include <cstddef>')
    lines.append('#include <cstdint>')
    lines.append('#include <complex>')
    lines.append('#include <vector>')
    lines.append('#include <memory>')
    lines.append('#include <algorithm>')
    lines.append('#include <cmath>')
    lines.append('#include <functional>')
    lines.append('#include <string>')
    lines.append('#include <stdexcept>')
    lines.append('#include <iostream>')
    lines.append('#include <type_traits>')
    lines.append('#include <cstring>')
    lines.append('#include <limits>')
    lines.append('#include <numeric>')
    lines.append('#include <utility>')
    lines.append('#include <random>')
    lines.append('#include <chrono>')
    lines.append('#include <sstream>')
    lines.append('#include <iomanip>')
    lines.append('#include <map>')
    lines.append('#include <optional>')
    lines.append('#include <stack>')
    lines.append('#include <unordered_map>')
    lines.append('#include <thread>')
    lines.append('')
    lines.append('#ifdef _OPENMP')
    lines.append('#include <omp.h>')
    lines.append('#endif')
    lines.append('')
    lines.append('// ============================================================================')
    lines.append('// Platform Detection for SIMD')
    lines.append('// ============================================================================')
    lines.append('')
    lines.append('#if defined(__AVX512F__) && defined(__AVX512DQ__)')
    lines.append('#include <immintrin.h>')
    lines.append('#define HAS_AVX512 1')
    lines.append('#define SIMD_WIDTH 8')
    lines.append('#define CACHE_LINE 64')
    lines.append('#elif defined(__AVX2__)')
    lines.append('#include <immintrin.h>')
    lines.append('#define HAS_AVX2 1')
    lines.append('#define SIMD_WIDTH 4')
    lines.append('#define CACHE_LINE 64')
    lines.append('#else')
    lines.append('#define SIMD_WIDTH 1')
    lines.append('#define CACHE_LINE 64')
    lines.append('#endif')
    lines.append('')
    lines.append('#if defined(__ARM_NEON) || defined(__ARM_NEON__)')
    lines.append('#include <arm_neon.h>')
    lines.append('#define HAS_NEON 1')
    lines.append('#if SIMD_WIDTH == 1')
    lines.append('#undef SIMD_WIDTH')
    lines.append('#define SIMD_WIDTH 2')
    lines.append('#endif')
    lines.append('#endif')
    lines.append('')
    lines.append('// ============================================================================')
    lines.append('// FFT Library (Notorious-FFT)')
    lines.append('// ============================================================================')
    lines.append('')
    lines.append('#ifdef CHARLTON_IMPLEMENTATION')
    lines.append('#define NOTORIOUS_FFT_IMPLEMENTATION')
    lines.append('#endif')
    lines.append('#include "notorious_fft.h"')
    lines.append('#include "notorious_fft.hpp"')
    lines.append('')
    lines.append('// ============================================================================')
    lines.append('// Static Assertions')
    lines.append('// ============================================================================')
    lines.append('')
    lines.append('static_assert(sizeof(std::complex<double>) == sizeof(double) * 2,')
    lines.append('              "std::complex<double> must be contiguous");')
    lines.append('static_assert(alignof(std::complex<double>) >= alignof(double),')
    lines.append('              "std::complex<double> alignment insufficient");')
    lines.append('')
    lines.append('#ifndef M_PI')
    lines.append('#define M_PI 3.14159265358979323846264338327950288')
    lines.append('#endif')
    lines.append('')
    lines.append('#ifndef M_PI_2')
    lines.append('#define M_PI_2 1.57079632679489661923132169163975144')
    lines.append('#endif')
    lines.append('')
    
    # Begin namespace
    lines.append('namespace charlton {')
    lines.append('')
    
    # Type aliases and constants (from core.hpp)
    lines.append('// ============================================================================')
    lines.append('// Type Aliases and Constants')
    lines.append('// ============================================================================')
    lines.append('')
    lines.append('template<typename Scalar>')
    lines.append('using Complex = std::complex<Scalar>;')
    lines.append('')
    lines.append('constexpr double DEFAULT_TOLERANCE = 1e-10;')
    lines.append('constexpr double MACHINE_EPSILON = std::numeric_limits<double>::epsilon();')
    lines.append('constexpr double CSD_EPSILON = 1e-100;')
    lines.append('')
    
    # Process each modular header
    for header in HEADER_ORDER:
        lines.append('')
        lines.append('// ============================================================================')
        lines.append(f'// Module: {header}')
        lines.append('// ============================================================================')
        lines.append('')
        
        content = extract_content(HEADER_DIR / header)
        if content:
            lines.append(content)
    
    # End namespace
    lines.append('')
    lines.append('} // namespace charlton')
    lines.append('')
    lines.append('#endif // CHARLTON_HPP')
    lines.append('')
    
    # Write output
    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Generated: {OUTPUT_FILE}")
    print(f"  Lines: {len(lines)}")
    print(f"  Size: {os.path.getsize(OUTPUT_FILE)} bytes")
    print(f"  Modules: {len(HEADER_ORDER)}")

if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    generate_amalgamation()
