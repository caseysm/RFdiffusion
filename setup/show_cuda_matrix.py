#!/usr/bin/env python3
"""
Quick command to display CUDA compatibility matrix
"""

import sys
from cuda_compatibility_matrix import CudaCompatibilityMatrix

def main():
    """Display CUDA compatibility matrix with optional filtering"""
    matrix = CudaCompatibilityMatrix()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Usage: python show_cuda_matrix.py [OPTIONS]")
            print("")
            print("Options:")
            print("  --system, -s      Show system-specific recommendations")
            print("  --full, -f        Show full compatibility matrix")
            print("  --modern, -m      Show only modern architectures (8.0+)")
            print("  --legacy, -l      Show legacy architectures")
            print("  --help, -h        Show this help")
            return
        
        elif sys.argv[1] in ["--system", "-s"]:
            # Show system-specific info
            recommendations = matrix.get_recommendations_for_system()
            
            if 'error' in recommendations:
                print(f"ERROR: {recommendations['error']}")
                return
            
            print("Your System Configuration:")
            print("=" * 50)
            print(f"GPU: {recommendations['gpu_name']}")
            print(f"CUDA Version: {recommendations['cuda_version']}")
            print(f"Compute Capability: {recommendations['compute_capability']}")
            print(f"Architecture: {recommendations['architecture']}")
            
            config = recommendations['config']
            
            # Color code the compatibility
            colors = {
                "optimal": "[OPTIMAL]",
                "good": "[GOOD]", 
                "minimum": "[MINIMUM]",
                "incompatible": "[INCOMPATIBLE]",
                "deprecated": "[DEPRECATED]"
            }
            color = colors.get(config.compatibility.value, "[UNKNOWN]")
            
            print(f"\n{color} Compatibility Assessment: {config.compatibility.value.upper()}")
            print(f"Recommended Configuration:")
            print(f"   Profile: {config.config_name}")
            print(f"   PyTorch: {config.pytorch_version}")
            print(f"   CUDA: {config.cuda_version}")
            print(f"   DGL: {config.dgl_version}")
            print(f"   Notes: {config.notes}")
            
            # Show relevant row from matrix
            print(f"\nCompatibility for Compute {recommendations['compute_capability']}:")
            print("-" * 60)
            cuda_versions = ["12.8", "12.1", "11.8", "11.6", "11.3", "11.0"]
            header = "CUDA Version  "
            for cuda_ver in cuda_versions:
                header += f"{cuda_ver:>8}"
            print(header)
            
            row = "Compatibility "
            for cuda_ver in cuda_versions:
                config_check = matrix.get_compatibility(recommendations['compute_capability'], cuda_ver)
                color_check = colors.get(config_check.compatibility.value, "[UNKNOWN]")
                row += f"{color_check:>8}"
            print(row)
            
        elif sys.argv[1] in ["--modern", "-m"]:
            # Show only modern architectures
            compute_caps = [10.0, 9.0, 8.9, 8.6, 8.0]
            cuda_versions = ["12.8", "12.1", "11.8", "11.6", "11.3"]
            print("Modern GPU Compatibility Matrix (Ampere & Newer)")
            print("=" * 70)
            matrix.display_compatibility_matrix(compute_caps, cuda_versions)
            
        elif sys.argv[1] in ["--legacy", "-l"]:
            # Show legacy architectures
            compute_caps = [7.5, 7.0, 6.1, 5.2]
            cuda_versions = ["12.1", "11.8", "11.6", "11.3", "11.0", "10.2"]
            print("Legacy GPU Compatibility Matrix")
            print("=" * 70)
            matrix.display_compatibility_matrix(compute_caps, cuda_versions)
            
        elif sys.argv[1] in ["--full", "-f"]:
            # Show full matrix
            matrix.display_compatibility_matrix()
            
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        # Default: show system info + modern matrix
        print("System Detection:")
        print("=" * 50)
        recommendations = matrix.get_recommendations_for_system()
        
        if 'error' in recommendations:
            print(f"ERROR: {recommendations['error']}")
        else:
            config = recommendations['config']
            colors = {
                "optimal": "[OPTIMAL]",
                "good": "[GOOD]", 
                "minimum": "[MINIMUM]",
                "incompatible": "[INCOMPATIBLE]",
                "deprecated": "[DEPRECATED]"
            }
            color = colors.get(config.compatibility.value, "[UNKNOWN]")
            
            print(f"GPU: {recommendations['gpu_name']}")
            print(f"Architecture: {recommendations['architecture']} (Compute {recommendations['compute_capability']})")
            print(f"CUDA: {recommendations['cuda_version']}")
            print(f"Status: {color} {config.compatibility.value.upper()}")
            print(f"Recommended: {config.config_name}")
        
        print("\nModern GPU Compatibility Matrix:")
        print("=" * 70)
        compute_caps = [9.0, 8.9, 8.6, 8.0, 7.5]
        cuda_versions = ["12.8", "12.1", "11.8", "11.6", "11.3"]
        matrix.display_compatibility_matrix(compute_caps, cuda_versions)
        
        print("\nUse --help for more options")

if __name__ == '__main__':
    main()