import sys
import torch
try:
    import triton
    print("--- Information Triton ---")
    print(f"1. File location (__file__): {getattr(triton, '__file__', 'None')}")
    print(f"2. Package path (__path__): {getattr(triton, '__path__', 'Not a package (Missing __path__)')}")
    print(f"3. Version: {getattr(triton, '__version__', 'Unknown')}")
    print(f"4. Python Path priority: {sys.path[0]}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"5. Device: {device}")
except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"Other Error: {e}")
    