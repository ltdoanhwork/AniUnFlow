
import sys
import os
from pathlib import Path

# Simulate running from project root
# Because sam_encoder logic depends on __file__ relative path which is correct regardless of cwd?
# sam_encoder logic: FILE.parents[2] -> models/aniunflow_v4 -> models -> AniUnFlow
# So it adds AniUnFlow/models/sam2 to sys.path

# Add project root to sys.path to find models module
sys.path.insert(0, os.getcwd())

print(f"Testing import from {os.getcwd()}")

try:
    from models.aniunflow_v4 import sam_encoder
    print("Imported sam_encoder successfully.")
    
    import sam2
    print(f"Imported sam2 from {sam2.__file__}")
    
    from sam2.build_sam import build_sam2
    print("Imported build_sam2 successfully.")
    
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
