import importlib
import traceback

pkgs = ['numpy','scipy','PIL','torch','torchvision','facenet_pytorch']
for p in pkgs:
    try:
        m = importlib.import_module(p)
        v = getattr(m, '__version__', None)
        print(f"{p} OK {v}")
    except Exception as e:
        print(f"{p} FAIL {type(e).__name__}: {e}")
        traceback.print_exc()

try:
    import torch
    print('cuda', torch.cuda.is_available())
except Exception as e:
    print('torch cuda check failed', e)
