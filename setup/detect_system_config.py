#!/usr/bin/env python3
"""
System Configuration Detection for RFdiffusion
Automatically detects GPU hardware and CUDA toolkit to recommend optimal package versions.
"""

import subprocess
import re
import json
import sys
from typing import Dict, Optional, Tuple, List
try:
    from cuda_compatibility_matrix import CudaCompatibilityMatrix
    MATRIX_AVAILABLE = True
except ImportError:
    MATRIX_AVAILABLE = False

class SystemDetector:
    def __init__(self):
        self.gpu_info = None
        self.cuda_info = None
        self.recommendations = {}
    
    def detect_nvidia_gpu(self) -> Optional[Dict]:
        """Detect NVIDIA GPU information using nvidia-smi with enhanced compute capability detection"""
        try:
            # Get GPU information
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=name,compute_cap,memory.total,driver_version', 
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        gpu_name = parts[0]
                        compute_cap_str = parts[1]
                        memory_mb = int(parts[2])
                        driver_version = parts[3]
                        
                        # Parse compute capability - handle different formats
                        compute_cap = self._parse_compute_capability(compute_cap_str, gpu_name)
                        
                        gpus.append({
                            'name': gpu_name,
                            'compute_capability': compute_cap,
                            'memory_gb': memory_mb // 1024,
                            'driver_version': driver_version
                        })
            
            return {'gpus': gpus, 'primary': gpus[0] if gpus else None}
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    
    def _parse_compute_capability(self, compute_cap_str: str, gpu_name: str) -> str:
        """Parse compute capability with fallback to GPU name lookup"""
        # Try direct parsing first
        try:
            if '.' in compute_cap_str:
                return compute_cap_str
            elif len(compute_cap_str) >= 2 and compute_cap_str.isdigit():
                # Handle format like "89" -> "8.9"
                return f"{compute_cap_str[0]}.{compute_cap_str[1:]}"
        except:
            pass
        
        # Fallback to GPU name lookup
        gpu_compute_map = {
            # Blackwell
            "NVIDIA GeForce RTX 5090": "10.0",
            "NVIDIA GeForce RTX 5080": "10.0", 
            "NVIDIA GeForce RTX 5070": "10.0",
            "NVIDIA GeForce RTX 5060": "10.0",
            "NVIDIA B100": "10.0",
            "NVIDIA B200": "10.0",
            "NVIDIA B300": "10.0",
            "NVIDIA GB200": "10.0",
            
            # Hopper
            "NVIDIA H100 PCIe": "9.0",
            "NVIDIA H100 SXM5": "9.0",
            "NVIDIA H100": "9.0",
            "NVIDIA H200": "9.0",
            "NVIDIA H200 SXM5": "9.0",
            
            # Ada Lovelace
            "NVIDIA GeForce RTX 4090": "8.9",
            "NVIDIA GeForce RTX 4080 SUPER": "8.9",
            "NVIDIA GeForce RTX 4080": "8.9",
            "NVIDIA GeForce RTX 4070 Ti SUPER": "8.9",
            "NVIDIA GeForce RTX 4070 Ti": "8.9",
            "NVIDIA GeForce RTX 4070 SUPER": "8.9",
            "NVIDIA GeForce RTX 4070": "8.9",
            "NVIDIA GeForce RTX 4060 Ti": "8.9",
            "NVIDIA GeForce RTX 4060": "8.9",
            
            # Ampere
            "NVIDIA A100-SXM4-80GB": "8.0",
            "NVIDIA A100-SXM4-40GB": "8.0",
            "NVIDIA A100-PCIE-80GB": "8.0",
            "NVIDIA A100-PCIE-40GB": "8.0",
            "NVIDIA A100": "8.0",
            "NVIDIA GeForce RTX 3090 Ti": "8.6",
            "NVIDIA GeForce RTX 3090": "8.6",
            "NVIDIA GeForce RTX 3080 Ti": "8.6",
            "NVIDIA GeForce RTX 3080": "8.6",
            "NVIDIA GeForce RTX 3070 Ti": "8.6",
            "NVIDIA GeForce RTX 3070": "8.6",
            "NVIDIA GeForce RTX 3060 Ti": "8.6",
            "NVIDIA GeForce RTX 3060": "8.6",
            "NVIDIA GeForce RTX 3050": "8.6",
            
            # Turing
            "NVIDIA GeForce RTX 2080 Ti": "7.5",
            "NVIDIA GeForce RTX 2080 SUPER": "7.5",
            "NVIDIA GeForce RTX 2080": "7.5",
            "NVIDIA GeForce RTX 2070 SUPER": "7.5",
            "NVIDIA GeForce RTX 2070": "7.5",
            "NVIDIA GeForce RTX 2060 SUPER": "7.5",
            "NVIDIA GeForce RTX 2060": "7.5",
            "NVIDIA GeForce GTX 1660 Ti": "7.5",
            "NVIDIA GeForce GTX 1660 SUPER": "7.5",
            "NVIDIA GeForce GTX 1660": "7.5",
            "NVIDIA GeForce GTX 1650": "7.5",
            
            # Volta
            "NVIDIA TITAN V": "7.0",
            "NVIDIA Tesla V100-SXM2-32GB": "7.0",
            "NVIDIA Tesla V100-SXM2-16GB": "7.0",
            "NVIDIA Tesla V100-PCIE-32GB": "7.0",
            "NVIDIA Tesla V100-PCIE-16GB": "7.0",
            "NVIDIA Tesla V100": "7.0",
            
            # Pascal
            "NVIDIA TITAN Xp": "6.1",
            "NVIDIA TITAN X (Pascal)": "6.1",
            "NVIDIA GeForce GTX 1080 Ti": "6.1",
            "NVIDIA GeForce GTX 1080": "6.1",
            "NVIDIA GeForce GTX 1070 Ti": "6.1",
            "NVIDIA GeForce GTX 1070": "6.1",
            "NVIDIA GeForce GTX 1060": "6.1",
            "NVIDIA GeForce GTX 1050 Ti": "6.1",
            "NVIDIA GeForce GTX 1050": "6.1",
            
            # Maxwell
            "NVIDIA GeForce GTX TITAN X": "5.2",
            "NVIDIA GeForce GTX 980 Ti": "5.2",
            "NVIDIA GeForce GTX 980": "5.2",
            "NVIDIA GeForce GTX 970": "5.2",
            "NVIDIA GeForce GTX 960": "5.2",
            "NVIDIA GeForce GTX 950": "5.2",
        }
        
        return gpu_compute_map.get(gpu_name, compute_cap_str)
    
    def detect_cuda_toolkit(self) -> Optional[Dict]:
        """Detect CUDA toolkit version"""
        cuda_info = {}
        
        # Try nvcc first
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, check=True)
            version_match = re.search(r'release (\d+\.\d+)', result.stdout)
            if version_match:
                cuda_info['nvcc_version'] = version_match.group(1)
        except (subprocess.CalledProcessError, FileNotFoundError):
            cuda_info['nvcc_version'] = None
        
        # Try nvidia-smi for runtime version
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
            version_match = re.search(r'CUDA Version: (\d+\.\d+)', result.stdout)
            if version_match:
                cuda_info['runtime_version'] = version_match.group(1)
        except (subprocess.CalledProcessError, FileNotFoundError):
            cuda_info['runtime_version'] = None
        
        # Check for conda CUDA packages
        try:
            result = subprocess.run(['conda', 'list', 'cuda'], capture_output=True, text=True)
            if result.returncode == 0:
                cuda_packages = []
                for line in result.stdout.split('\n'):
                    if 'cuda' in line.lower() and not line.startswith('#'):
                        cuda_packages.append(line.strip())
                cuda_info['conda_packages'] = cuda_packages
        except (subprocess.CalledProcessError, FileNotFoundError):
            cuda_info['conda_packages'] = []
        
        return cuda_info if any(cuda_info.values()) else None
    
    def get_pytorch_recommendation(self, gpu_info: Dict, cuda_info: Dict) -> Dict:
        """Recommend PyTorch version based on hardware"""
        recommendations = {}
        
        if not gpu_info or not gpu_info.get('primary'):
            recommendations['config'] = 'cpu-only'
            recommendations['pytorch_version'] = '2.7.0'
            recommendations['install_url'] = 'https://download.pytorch.org/whl/cpu'
            return recommendations
        
        primary_gpu = gpu_info['primary']
        compute_cap = primary_gpu.get('compute_capability', '0.0')
        gpu_memory = primary_gpu.get('memory_gb', 0)
        gpu_name = primary_gpu.get('name', '').lower()
        
        # Parse compute capability
        try:
            major, minor = map(int, compute_cap.split('.'))
            compute_num = major * 10 + minor
        except:
            compute_num = 0
        
        # CUDA version detection priority: nvcc > runtime
        cuda_version = cuda_info.get('nvcc_version') or cuda_info.get('runtime_version')
        
        # Determine optimal configuration  
        if compute_num >= 100:  # Blackwell RTX 50xx series (10.0) - requires CUDA 12.8+
            if cuda_version and float(cuda_version) >= 12.8:
                recommendations['config'] = 'bleeding-edge'
                recommendations['pytorch_version'] = '2.7.0'
                recommendations['cuda_version'] = '12.8'
                recommendations['dgl_version'] = '2.4+'
                recommendations['install_url'] = 'https://download.pytorch.org/whl/cu128'
                recommendations['reason'] = 'Blackwell architecture requires CUDA 12.8+ - optimal configuration'
            else:
                recommendations['config'] = 'incompatible'
                recommendations['error'] = f'Blackwell GPUs require CUDA 12.8+ (detected: {cuda_version or "none"})'
                recommendations['reason'] = 'Compute capability 10.0+ needs CUDA 12.8+ for native compilation'
        
        elif compute_num >= 89:  # H100/H200 (9.0), RTX 40xx series (8.9) - Ada Lovelace, Hopper
            # H100/H200 (Hopper 9.0) and RTX 40xx (Ada Lovelace 8.9) require CUDA 11.8+ minimum
            if cuda_version and float(cuda_version) >= 12.8:
                recommendations['config'] = 'bleeding-edge'
                recommendations['pytorch_version'] = '2.7.0'
                recommendations['cuda_version'] = '12.8'
                recommendations['dgl_version'] = '2.4+'
                recommendations['install_url'] = 'https://download.pytorch.org/whl/cu128'
                recommendations['reason'] = 'Latest hardware with CUDA 12.8+ - optimal for Ada Lovelace/Hopper'
            elif cuda_version and float(cuda_version) >= 12.1:
                recommendations['config'] = 'pytorch27-cuda121'
                recommendations['pytorch_version'] = '2.7.0'
                recommendations['cuda_version'] = '12.1'
                recommendations['dgl_version'] = '2.1.0'
                recommendations['install_url'] = 'https://download.pytorch.org/whl/cu121'
                recommendations['reason'] = 'Latest hardware with CUDA 12.1+ - good performance'
            elif cuda_version and float(cuda_version) >= 11.8:
                recommendations['config'] = 'stable'
                recommendations['pytorch_version'] = '1.12.1'
                recommendations['cuda_version'] = '11.8'
                recommendations['dgl_version'] = '1.1.3'
                recommendations['install_url'] = 'https://download.pytorch.org/whl/cu118'
                recommendations['reason'] = 'Minimum CUDA 11.8+ required for Hopper/Ada Lovelace (compute capability 8.9+)'
            else:
                recommendations['config'] = 'incompatible'
                recommendations['error'] = f'H100/H200/RTX40xx requires CUDA 11.8+ (detected: {cuda_version or "none"})'
                recommendations['reason'] = 'Compute capability 8.9+ needs CUDA 11.8+ for native compilation'
        
        elif compute_num >= 80:  # A100, RTX 30xx series (Ampere)
            if cuda_version and float(cuda_version) >= 12.8:
                recommendations['config'] = 'bleeding-edge'
                recommendations['pytorch_version'] = '2.7.0'
                recommendations['cuda_version'] = '12.8'
                recommendations['dgl_version'] = '2.4+'
                recommendations['install_url'] = 'https://download.pytorch.org/whl/cu128'
                recommendations['reason'] = 'Ampere architecture with latest CUDA - excellent performance'
            elif cuda_version and float(cuda_version) >= 12.1:
                recommendations['config'] = 'pytorch27-cuda121'
                recommendations['pytorch_version'] = '2.7.0'
                recommendations['cuda_version'] = '12.1'
                recommendations['dgl_version'] = '2.1.0'
                recommendations['install_url'] = 'https://download.pytorch.org/whl/cu121'
                recommendations['reason'] = 'Ampere architecture - well supported'
            else:
                recommendations['config'] = 'stable'
                recommendations['pytorch_version'] = '1.12.1'
                recommendations['cuda_version'] = '11.6'
                recommendations['dgl_version'] = '1.1.3'
                recommendations['install_url'] = 'https://download.pytorch.org/whl/cu116'
                recommendations['reason'] = 'Stable configuration for Ampere'
        
        elif compute_num >= 75:  # RTX 20xx series (Turing)
            if cuda_version and float(cuda_version) >= 12.1:
                recommendations['config'] = 'pytorch27-cuda121'
                recommendations['pytorch_version'] = '2.7.0' 
                recommendations['cuda_version'] = '12.1'
                recommendations['dgl_version'] = '2.1.0'
                recommendations['install_url'] = 'https://download.pytorch.org/whl/cu121'
                recommendations['reason'] = 'Turing architecture with modern CUDA'
            else:
                recommendations['config'] = 'stable'
                recommendations['pytorch_version'] = '1.12.1'
                recommendations['cuda_version'] = '11.6'
                recommendations['dgl_version'] = '1.1.3'
                recommendations['install_url'] = 'https://download.pytorch.org/whl/cu116'
                recommendations['reason'] = 'Stable configuration for Turing'
        
        elif compute_num >= 70:  # V100, RTX 10xx series
            recommendations['config'] = 'stable'
            recommendations['pytorch_version'] = '1.12.1'
            recommendations['cuda_version'] = '11.6'
            recommendations['dgl_version'] = '1.1.3'
            recommendations['install_url'] = 'https://download.pytorch.org/whl/cu116'
            recommendations['reason'] = 'Stable configuration for Pascal/Volta'
        
        else:  # Older hardware
            recommendations['config'] = 'legacy'
            recommendations['pytorch_version'] = '1.12.1'
            recommendations['cuda_version'] = '11.3'
            recommendations['dgl_version'] = '1.1.3'
            recommendations['install_url'] = 'https://download.pytorch.org/whl/cu113'
            recommendations['reason'] = 'Legacy configuration for older hardware'
        
        # Memory warnings
        if gpu_memory < 8:
            recommendations['memory_warning'] = f'GPU has only {gpu_memory}GB memory - may need to reduce batch sizes'
        elif gpu_memory >= 24:
            recommendations['memory_note'] = f'GPU has {gpu_memory}GB memory - can handle large batch sizes'
        
        return recommendations
    
    def detect_system(self) -> Dict:
        """Perform complete system detection"""
        print("Detecting system configuration...")
        
        self.gpu_info = self.detect_nvidia_gpu()
        self.cuda_info = self.detect_cuda_toolkit()
        
        if self.gpu_info:
            print(f"Found {len(self.gpu_info['gpus'])} NVIDIA GPU(s)")
            for i, gpu in enumerate(self.gpu_info['gpus']):
                print(f"   GPU {i}: {gpu['name']} (Compute {gpu['compute_capability']}, {gpu['memory_gb']}GB)")
        else:
            print("No NVIDIA GPUs detected")
        
        if self.cuda_info:
            if self.cuda_info.get('nvcc_version'):
                print(f"CUDA Toolkit: {self.cuda_info['nvcc_version']} (nvcc)")
            if self.cuda_info.get('runtime_version'):
                print(f"CUDA Runtime: {self.cuda_info['runtime_version']} (nvidia-smi)")
        else:
            print("No CUDA installation detected")
        
        # Use compatibility matrix if available, fallback to legacy logic
        if MATRIX_AVAILABLE:
            self.recommendations = self._get_matrix_recommendations()
        else:
            self.recommendations = self.get_pytorch_recommendation(self.gpu_info or {}, self.cuda_info or {})
        
        return {
            'gpu_info': self.gpu_info,
            'cuda_info': self.cuda_info,
            'recommendations': self.recommendations
        }
    
    def _get_matrix_recommendations(self) -> Dict:
        """Get recommendations using the compatibility matrix"""
        try:
            matrix = CudaCompatibilityMatrix()
            
            gpu_name = None
            cuda_version = None
            
            if self.gpu_info and self.gpu_info.get('primary'):
                gpu_name = self.gpu_info['primary']['name']
            
            if self.cuda_info:
                cuda_version = self.cuda_info.get('nvcc_version') or self.cuda_info.get('runtime_version')
            
            matrix_result = matrix.get_recommendations_for_system(gpu_name, cuda_version)
            
            if 'error' in matrix_result:
                # Fallback to legacy logic
                return self.get_pytorch_recommendation(self.gpu_info or {}, self.cuda_info or {})
            
            config = matrix_result['config']
            
            # Convert matrix result to legacy format
            recommendations = {
                'config': config.config_name,
                'pytorch_version': config.pytorch_version,
                'cuda_version': config.cuda_version,
                'dgl_version': config.dgl_version,
                'install_url': config.install_url,
                'compatibility_level': config.compatibility.value,
                'reason': config.notes,
                'compute_capability': matrix_result.get('compute_capability'),
                'architecture': matrix_result.get('architecture')
            }
            
            # Add memory considerations
            if self.gpu_info and self.gpu_info.get('primary'):
                gpu_memory = self.gpu_info['primary'].get('memory_gb', 0)
                if gpu_memory < 8:
                    recommendations['memory_warning'] = f'GPU has only {gpu_memory}GB memory - may need to reduce batch sizes'
                elif gpu_memory >= 24:
                    recommendations['memory_note'] = f'GPU has {gpu_memory}GB memory - can handle large batch sizes'
            
            return recommendations
            
        except Exception as e:
            print(f"Warning: Matrix lookup failed ({e}), using legacy detection")
            return self.get_pytorch_recommendation(self.gpu_info or {}, self.cuda_info or {})
    
    def print_recommendations(self):
        """Print formatted recommendations"""
        if not self.recommendations:
            print("No recommendations available")
            return
        
        if self.recommendations.get('config') == 'incompatible':
            print("\nHardware Incompatibility Detected:")
            print(f"   Error: {self.recommendations.get('error', 'Unknown error')}")
            print(f"   Reason: {self.recommendations.get('reason', 'N/A')}")
            return
        
        print("\nRecommended Configuration:")
        print(f"   Config Profile: {self.recommendations.get('config', 'unknown')}")
        print(f"   PyTorch: {self.recommendations.get('pytorch_version', 'unknown')}")
        print(f"   CUDA: {self.recommendations.get('cuda_version', 'unknown')}")
        print(f"   DGL: {self.recommendations.get('dgl_version', 'unknown')}")
        
        # Show compatibility level if available
        if 'compatibility_level' in self.recommendations:
            colors = {"optimal": "[OPTIMAL]", "good": "[GOOD]", "minimum": "[MINIMUM]", "incompatible": "[INCOMPATIBLE]", "deprecated": "[DEPRECATED]"}
            level = self.recommendations['compatibility_level']
            color = colors.get(level, "[UNKNOWN]")
            print(f"   Compatibility: {color} {level.upper()}")
        
        # Show architecture if available
        if 'architecture' in self.recommendations and 'compute_capability' in self.recommendations:
            print(f"   Architecture: {self.recommendations['architecture']} (Compute {self.recommendations['compute_capability']})")
        
        print(f"   Reason: {self.recommendations.get('reason', 'N/A')}")
        
        if 'memory_warning' in self.recommendations:
            print(f"WARNING: {self.recommendations['memory_warning']}")
        elif 'memory_note' in self.recommendations:
            print(f"INFO: {self.recommendations['memory_note']}")
    
    def export_config(self, output_file: str = None):
        """Export detection results to JSON"""
        config = {
            'detection_results': {
                'gpu_info': self.gpu_info,
                'cuda_info': self.cuda_info,
                'recommendations': self.recommendations
            }
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Configuration exported to {output_file}")
        
        return config

def main():
    detector = SystemDetector()
    results = detector.detect_system()
    detector.print_recommendations()
    
    # Export for use by setup script
    detector.export_config('system_config.json')
    
    return results

if __name__ == '__main__':
    main()