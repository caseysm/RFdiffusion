#!/usr/bin/env python3
"""
CUDA Compatibility Matrix for RFdiffusion
Comprehensive mapping of compute capabilities, CUDA versions, and optimal configurations
"""

import subprocess
import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, replace
from enum import Enum

class CompatibilityLevel(Enum):
    OPTIMAL = "optimal"
    GOOD = "good"
    MINIMUM = "minimum"
    INCOMPATIBLE = "incompatible"
    DEPRECATED = "deprecated"

@dataclass
class CudaConfig:
    pytorch_version: str
    cuda_version: str
    dgl_version: str
    install_url: str
    config_name: str
    compatibility: CompatibilityLevel
    notes: str = ""

class CudaCompatibilityMatrix:
    def __init__(self):
        self.matrix = self._build_compatibility_matrix()
        self.compute_capabilities = self._get_compute_capability_map()
    
    def _build_compatibility_matrix(self) -> Dict[Tuple[float, str], CudaConfig]:
        """Build comprehensive compatibility matrix"""
        matrix = {}
        
        # Define configurations
        bleeding_edge = CudaConfig("2.7.0", "12.8", "2.4+", "https://download.pytorch.org/whl/cu128", "bleeding-edge", CompatibilityLevel.OPTIMAL)
        pytorch27_cuda121 = CudaConfig("2.7.0", "12.1", "2.1.0", "https://download.pytorch.org/whl/cu121", "pytorch27-cuda121", CompatibilityLevel.GOOD)
        stable_118 = CudaConfig("1.12.1", "11.8", "1.1.3", "https://download.pytorch.org/whl/cu118", "stable", CompatibilityLevel.MINIMUM)
        stable_116 = CudaConfig("1.12.1", "11.6", "1.1.3", "https://download.pytorch.org/whl/cu116", "stable", CompatibilityLevel.GOOD)
        stable_113 = CudaConfig("1.12.1", "11.3", "1.1.3", "https://download.pytorch.org/whl/cu113", "legacy", CompatibilityLevel.MINIMUM)
        cpu_only = CudaConfig("2.7.0", "cpu", "2.4+", "https://download.pytorch.org/whl/cpu", "cpu-only", CompatibilityLevel.GOOD)
        incompatible = CudaConfig("", "", "", "", "incompatible", CompatibilityLevel.INCOMPATIBLE)
        
        # CUDA versions to check
        cuda_versions = ["12.9", "12.8", "12.7", "12.6", "12.5", "12.4", "12.3", "12.2", "12.1", "12.0", 
                        "11.8", "11.7", "11.6", "11.5", "11.4", "11.3", "11.2", "11.1", "11.0", "10.2", "10.1", "10.0"]
        
        # Blackwell RTX 50xx, B-series (10.0) - Requires CUDA 12.8+
        for cuda_ver in cuda_versions:
            cuda_float = float(cuda_ver)
            if cuda_float >= 12.8:
                matrix[(10.0, cuda_ver)] = replace(bleeding_edge, notes="Optimal for Blackwell architecture")
            else:
                matrix[(10.0, cuda_ver)] = replace(incompatible, notes=f"Compute 10.0 requires CUDA 12.8+")
        
        # Hopper H100, H200 (9.0) - Requires CUDA 11.8+
        for cuda_ver in cuda_versions:
            cuda_float = float(cuda_ver)
            if cuda_float >= 12.8:
                matrix[(9.0, cuda_ver)] = replace(bleeding_edge, notes="Optimal for Hopper architecture")
            elif cuda_float >= 12.1:
                matrix[(9.0, cuda_ver)] = replace(pytorch27_cuda121, notes="Good performance for Hopper")
            elif cuda_float >= 11.8:
                matrix[(9.0, cuda_ver)] = replace(stable_118, notes="Minimum CUDA for compute 9.0")
            else:
                matrix[(9.0, cuda_ver)] = replace(incompatible, notes=f"Compute 9.0 requires CUDA 11.8+")
        
        # Ada Lovelace RTX 40xx (8.9) - Requires CUDA 11.8+
        for cuda_ver in cuda_versions:
            cuda_float = float(cuda_ver)
            if cuda_float >= 12.8:
                matrix[(8.9, cuda_ver)] = replace(bleeding_edge, notes="Optimal for Ada Lovelace")
            elif cuda_float >= 12.1:
                matrix[(8.9, cuda_ver)] = replace(pytorch27_cuda121, notes="Good performance for Ada Lovelace")
            elif cuda_float >= 11.8:
                matrix[(8.9, cuda_ver)] = replace(stable_118, notes="Minimum CUDA for compute 8.9")
            else:
                matrix[(8.9, cuda_ver)] = replace(incompatible, notes=f"Compute 8.9 requires CUDA 11.8+")
        
        # Ampere A100 (8.0), RTX 30xx (8.6) - Works with CUDA 11.0+
        for compute_cap in [8.0, 8.6]:
            for cuda_ver in cuda_versions:
                cuda_float = float(cuda_ver)
                if cuda_float >= 12.8:
                    matrix[(compute_cap, cuda_ver)] = replace(bleeding_edge, notes="Excellent for Ampere")
                elif cuda_float >= 12.1:
                    matrix[(compute_cap, cuda_ver)] = replace(pytorch27_cuda121, notes="Very good for Ampere")
                elif cuda_float >= 11.6:
                    matrix[(compute_cap, cuda_ver)] = replace(stable_116, notes="Good for Ampere")
                elif cuda_float >= 11.0:
                    matrix[(compute_cap, cuda_ver)] = replace(stable_113, notes="Basic support for Ampere")
                else:
                    matrix[(compute_cap, cuda_ver)] = replace(incompatible, notes=f"CUDA too old for compute {compute_cap}")
        
        # Turing RTX 20xx (7.5) - Works with CUDA 10.0+
        for cuda_ver in cuda_versions:
            cuda_float = float(cuda_ver)
            if cuda_float >= 12.8:
                matrix[(7.5, cuda_ver)] = replace(bleeding_edge, notes="Excellent for Turing")
            elif cuda_float >= 12.1:
                matrix[(7.5, cuda_ver)] = replace(pytorch27_cuda121, notes="Very good for Turing")
            elif cuda_float >= 11.6:
                matrix[(7.5, cuda_ver)] = replace(stable_116, notes="Good for Turing")
            elif cuda_float >= 10.0:
                matrix[(7.5, cuda_ver)] = replace(stable_113, notes="Basic support for Turing")
            else:
                matrix[(7.5, cuda_ver)] = replace(incompatible, notes=f"CUDA too old for compute 7.5")
        
        # Volta V100 (7.0) - Works with CUDA 9.0+
        for cuda_ver in cuda_versions:
            cuda_float = float(cuda_ver)
            if cuda_float >= 12.1:
                matrix[(7.0, cuda_ver)] = replace(pytorch27_cuda121, notes="Good for Volta")
            elif cuda_float >= 11.6:
                matrix[(7.0, cuda_ver)] = replace(stable_116, notes="Recommended for Volta")
            elif cuda_float >= 10.0:
                matrix[(7.0, cuda_ver)] = replace(stable_113, notes="Basic support for Volta")
            else:
                matrix[(7.0, cuda_ver)] = replace(incompatible, notes=f"CUDA too old for compute 7.0")
        
        # Pascal GTX 10xx (6.1) - Works with CUDA 8.0+
        for cuda_ver in cuda_versions:
            cuda_float = float(cuda_ver)
            if cuda_float >= 11.6:
                matrix[(6.1, cuda_ver)] = replace(stable_116, notes="Good for Pascal")
            elif cuda_float >= 10.0:
                matrix[(6.1, cuda_ver)] = replace(stable_113, notes="Recommended for Pascal")
            else:
                matrix[(6.1, cuda_ver)] = replace(incompatible, notes=f"CUDA too old for compute 6.1")
        
        # Maxwell GTX 9xx (5.2) - Legacy support
        for cuda_ver in cuda_versions:
            cuda_float = float(cuda_ver)
            if cuda_float >= 11.0:
                matrix[(5.2, cuda_ver)] = replace(stable_113, compatibility=CompatibilityLevel.DEPRECATED, notes="Legacy Maxwell support")
            elif cuda_float >= 10.0:
                matrix[(5.2, cuda_ver)] = replace(stable_113, compatibility=CompatibilityLevel.MINIMUM, notes="Basic Maxwell support")
            else:
                matrix[(5.2, cuda_ver)] = replace(incompatible, notes=f"CUDA too old for compute 5.2")
        
        return matrix
    
    def _get_compute_capability_map(self) -> Dict[str, float]:
        """Map GPU names to compute capabilities"""
        return {
            # Blackwell (RTX 50xx series, B-series data center GPUs)
            "NVIDIA GeForce RTX 5090": 10.0,
            "NVIDIA GeForce RTX 5080": 10.0,
            "NVIDIA GeForce RTX 5070": 10.0,
            "NVIDIA GeForce RTX 5060": 10.0,
            "NVIDIA B100": 10.0,
            "NVIDIA B200": 10.0,
            "NVIDIA B300": 10.0,
            "NVIDIA GB200": 10.0,
            
            # Hopper (H100, H200)
            "NVIDIA H100 PCIe": 9.0,
            "NVIDIA H100 SXM5": 9.0,
            "NVIDIA H100": 9.0,
            "NVIDIA H200": 9.0,
            "NVIDIA H200 SXM5": 9.0,
            
            # Ada Lovelace (RTX 40xx)
            "NVIDIA GeForce RTX 4090": 8.9,
            "NVIDIA GeForce RTX 4080 SUPER": 8.9,
            "NVIDIA GeForce RTX 4080": 8.9,
            "NVIDIA GeForce RTX 4070 Ti SUPER": 8.9,
            "NVIDIA GeForce RTX 4070 Ti": 8.9,
            "NVIDIA GeForce RTX 4070 SUPER": 8.9,
            "NVIDIA GeForce RTX 4070": 8.9,
            "NVIDIA GeForce RTX 4060 Ti": 8.9,
            "NVIDIA GeForce RTX 4060": 8.9,
            "NVIDIA RTX 6000 Ada Generation": 8.9,
            "NVIDIA RTX 5000 Ada Generation": 8.9,
            "NVIDIA RTX 4000 Ada Generation": 8.9,
            "NVIDIA L40S": 8.9,
            "NVIDIA L40": 8.9,
            "NVIDIA L4": 8.9,
            
            # Ampere (RTX 30xx, A100)
            "NVIDIA A100-SXM4-80GB": 8.0,
            "NVIDIA A100-SXM4-40GB": 8.0,
            "NVIDIA A100-PCIE-80GB": 8.0,
            "NVIDIA A100-PCIE-40GB": 8.0,
            "NVIDIA A100": 8.0,
            "NVIDIA GeForce RTX 3090 Ti": 8.6,
            "NVIDIA GeForce RTX 3090": 8.6,
            "NVIDIA GeForce RTX 3080 Ti": 8.6,
            "NVIDIA GeForce RTX 3080": 8.6,
            "NVIDIA GeForce RTX 3070 Ti": 8.6,
            "NVIDIA GeForce RTX 3070": 8.6,
            "NVIDIA GeForce RTX 3060 Ti": 8.6,
            "NVIDIA GeForce RTX 3060": 8.6,
            "NVIDIA GeForce RTX 3050": 8.6,
            "NVIDIA RTX A6000": 8.6,
            "NVIDIA RTX A5000": 8.6,
            "NVIDIA RTX A4000": 8.6,
            
            # Turing (RTX 20xx, GTX 16xx)
            "NVIDIA GeForce RTX 2080 Ti": 7.5,
            "NVIDIA GeForce RTX 2080 SUPER": 7.5,
            "NVIDIA GeForce RTX 2080": 7.5,
            "NVIDIA GeForce RTX 2070 SUPER": 7.5,
            "NVIDIA GeForce RTX 2070": 7.5,
            "NVIDIA GeForce RTX 2060 SUPER": 7.5,
            "NVIDIA GeForce RTX 2060": 7.5,
            "NVIDIA GeForce GTX 1660 Ti": 7.5,
            "NVIDIA GeForce GTX 1660 SUPER": 7.5,
            "NVIDIA GeForce GTX 1660": 7.5,
            "NVIDIA GeForce GTX 1650": 7.5,
            "NVIDIA Quadro RTX 8000": 7.5,
            "NVIDIA Quadro RTX 6000": 7.5,
            "NVIDIA Quadro RTX 5000": 7.5,
            "NVIDIA Quadro RTX 4000": 7.5,
            
            # Volta
            "NVIDIA TITAN V": 7.0,
            "NVIDIA Tesla V100-SXM2-32GB": 7.0,
            "NVIDIA Tesla V100-SXM2-16GB": 7.0,
            "NVIDIA Tesla V100-PCIE-32GB": 7.0,
            "NVIDIA Tesla V100-PCIE-16GB": 7.0,
            "NVIDIA Tesla V100": 7.0,
            "NVIDIA Quadro GV100": 7.0,
            
            # Pascal (GTX 10xx)
            "NVIDIA TITAN Xp": 6.1,
            "NVIDIA TITAN X (Pascal)": 6.1,
            "NVIDIA GeForce GTX 1080 Ti": 6.1,
            "NVIDIA GeForce GTX 1080": 6.1,
            "NVIDIA GeForce GTX 1070 Ti": 6.1,
            "NVIDIA GeForce GTX 1070": 6.1,
            "NVIDIA GeForce GTX 1060": 6.1,
            "NVIDIA GeForce GTX 1050 Ti": 6.1,
            "NVIDIA GeForce GTX 1050": 6.1,
            "NVIDIA Tesla P100": 6.0,
            "NVIDIA Tesla P40": 6.1,
            "NVIDIA Tesla P4": 6.1,
            "NVIDIA Quadro P6000": 6.1,
            "NVIDIA Quadro P5000": 6.1,
            "NVIDIA Quadro P4000": 6.1,
            
            # Maxwell (GTX 9xx)
            "NVIDIA GeForce GTX TITAN X": 5.2,
            "NVIDIA GeForce GTX 980 Ti": 5.2,
            "NVIDIA GeForce GTX 980": 5.2,
            "NVIDIA GeForce GTX 970": 5.2,
            "NVIDIA GeForce GTX 960": 5.2,
            "NVIDIA GeForce GTX 950": 5.2,
            "NVIDIA Tesla M60": 5.2,
            "NVIDIA Tesla M40": 5.2,
            "NVIDIA Quadro M6000": 5.2,
        }
    
    def auto_detect_compute_capability(self) -> Optional[float]:
        """Auto-detect compute capability from nvidia-smi"""
        try:
            # Try nvidia-ml-py first (more accurate)
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=name,compute_cap', 
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)
            
            lines = result.stdout.strip().split('\n')
            if lines and lines[0].strip():
                parts = [p.strip() for p in lines[0].split(',')]
                if len(parts) >= 2:
                    gpu_name = parts[0]
                    compute_cap_str = parts[1]
                    
                    # Parse compute capability
                    try:
                        compute_cap = float(compute_cap_str)
                        return compute_cap
                    except ValueError:
                        pass
                    
                    # Fallback to name lookup
                    return self.compute_capabilities.get(gpu_name)
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Fallback to deviceQuery if available
        try:
            result = subprocess.run(['deviceQuery'], capture_output=True, text=True, check=True)
            
            # Parse deviceQuery output
            for line in result.stdout.split('\n'):
                if 'CUDA Capability Major/Minor version number' in line:
                    match = re.search(r'(\d+)\.(\d+)', line)
                    if match:
                        major, minor = match.groups()
                        return float(f"{major}.{minor}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        return None
    
    def get_compatibility(self, compute_capability: float, cuda_version: str) -> CudaConfig:
        """Get compatibility for specific compute capability and CUDA version"""
        key = (compute_capability, cuda_version)
        
        if key in self.matrix:
            return self.matrix[key]
        
        # Find closest match
        best_match = None
        best_score = float('inf')
        
        for (cap, ver), config in self.matrix.items():
            if abs(cap - compute_capability) < 0.1:  # Same compute capability
                try:
                    ver_diff = abs(float(ver) - float(cuda_version))
                    if ver_diff < best_score:
                        best_score = ver_diff
                        best_match = config
                except ValueError:
                    continue
        
        if best_match:
            return replace(best_match, notes=f"Approximate match for compute {compute_capability}, CUDA {cuda_version}")
        
        # Default fallback
        return CudaConfig("", "", "", "", "unknown", CompatibilityLevel.INCOMPATIBLE, 
                         f"No compatibility data for compute {compute_capability}, CUDA {cuda_version}")
    
    def display_compatibility_matrix(self, compute_caps: List[float] = None, cuda_versions: List[str] = None):
        """Display visual compatibility matrix"""
        if compute_caps is None:
            compute_caps = [9.0, 8.9, 8.6, 8.0, 7.5, 7.0, 6.1, 5.2]
        
        if cuda_versions is None:
            cuda_versions = ["12.8", "12.1", "11.8", "11.6", "11.3", "11.0", "10.2"]
        
        # Color coding
        colors = {
            CompatibilityLevel.OPTIMAL: "[OPTIMAL]",
            CompatibilityLevel.GOOD: "[GOOD]", 
            CompatibilityLevel.MINIMUM: "[MINIMUM]",
            CompatibilityLevel.INCOMPATIBLE: "[INCOMPATIBLE]",
            CompatibilityLevel.DEPRECATED: "[DEPRECATED]"
        }
        
        print("\nCUDA Compatibility Matrix")
        print("=" * 80)
        
        # Header
        header = "Compute Cap  "
        for cuda_ver in cuda_versions:
            header += f"CUDA {cuda_ver:>6}"
        print(header)
        print("-" * len(header))
        
        # Matrix rows
        for compute_cap in compute_caps:
            row = f"{compute_cap:>6.1f}      "
            for cuda_ver in cuda_versions:
                config = self.get_compatibility(compute_cap, cuda_ver)
                color = colors.get(config.compatibility, "[UNKNOWN]")
                row += f"  {color:>6}"
            
            # Add architecture name
            arch_names = {
                10.0: " (Blackwell RTX50xx)",
                9.0: " (Hopper H100/H200)",
                8.9: " (Ada Lovelace RTX40xx)",
                8.6: " (Ampere RTX30xx)",
                8.0: " (Ampere A100)",
                7.5: " (Turing RTX20xx)",
                7.0: " (Volta V100)",
                6.1: " (Pascal GTX10xx)",
                5.2: " (Maxwell GTX9xx)"
            }
            row += arch_names.get(compute_cap, "")
            print(row)
        
        # Legend
        print("\nLegend:")
        print("[OPTIMAL] Optimal   [GOOD] Good   [MINIMUM] Minimum   [INCOMPATIBLE] Incompatible   [DEPRECATED] Deprecated")
    
    def get_recommendations_for_system(self, gpu_name: str = None, cuda_version: str = None) -> Dict:
        """Get recommendations for current system"""
        # Auto-detect if not provided
        if not gpu_name or not cuda_version:
            try:
                result = subprocess.run([
                    'nvidia-smi', '--query-gpu=name', '--format=csv,noheader'
                ], capture_output=True, text=True, check=True)
                detected_gpu = result.stdout.strip()
                if detected_gpu and not gpu_name:
                    gpu_name = detected_gpu
            except:
                pass
            
            if not cuda_version:
                try:
                    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
                    version_match = re.search(r'CUDA Version: (\d+\.\d+)', result.stdout)
                    if version_match:
                        cuda_version = version_match.group(1)
                except:
                    pass
        
        # Get compute capability
        compute_cap = None
        if gpu_name:
            compute_cap = self.compute_capabilities.get(gpu_name)
        
        if not compute_cap:
            compute_cap = self.auto_detect_compute_capability()
        
        if not compute_cap or not cuda_version:
            return {
                'error': 'Unable to detect GPU or CUDA version',
                'gpu_name': gpu_name,
                'cuda_version': cuda_version,
                'compute_capability': compute_cap
            }
        
        # Get compatibility
        config = self.get_compatibility(compute_cap, cuda_version)
        
        return {
            'gpu_name': gpu_name,
            'cuda_version': cuda_version,
            'compute_capability': compute_cap,
            'config': config,
            'architecture': self._get_architecture_name(compute_cap)
        }
    
    def _get_architecture_name(self, compute_cap: float) -> str:
        """Get architecture name from compute capability"""
        if compute_cap >= 10.0:
            return "Blackwell"
        elif compute_cap >= 9.0:
            return "Hopper"
        elif compute_cap >= 8.9:
            return "Ada Lovelace"
        elif compute_cap >= 8.0:
            return "Ampere"
        elif compute_cap >= 7.0:
            return "Volta"
        elif compute_cap >= 6.0:
            return "Pascal"
        elif compute_cap >= 5.0:
            return "Maxwell"
        elif compute_cap >= 3.0:
            return "Kepler"
        else:
            return "Legacy"

def main():
    matrix = CudaCompatibilityMatrix()
    
    # Display matrix
    matrix.display_compatibility_matrix()
    
    # Get system recommendations
    print("\nSystem Detection:")
    print("=" * 50)
    recommendations = matrix.get_recommendations_for_system()
    
    if 'error' in recommendations:
        print(f"ERROR: {recommendations['error']}")
    else:
        print(f"GPU: {recommendations['gpu_name']}")
        print(f"CUDA Version: {recommendations['cuda_version']}")
        print(f"Compute Capability: {recommendations['compute_capability']}")
        print(f"Architecture: {recommendations['architecture']}")
        
        config = recommendations['config']
        print(f"\nRecommended Configuration:")
        print(f"  Profile: {config.config_name}")
        print(f"  PyTorch: {config.pytorch_version}")
        print(f"  CUDA: {config.cuda_version}")
        print(f"  DGL: {config.dgl_version}")
        print(f"  Compatibility: {config.compatibility.value}")
        print(f"  Notes: {config.notes}")

if __name__ == '__main__':
    main()