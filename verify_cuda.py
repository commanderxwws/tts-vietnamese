import torch
import sys

def check_cuda():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"Device count: {device_count}")
        
        for i in range(device_count):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            
        print("\nTesting CUDA tensor operation...")
        try:
            x = torch.tensor([1.0, 2.0, 3.0]).cuda()
            y = torch.tensor([4.0, 5.0, 6.0]).cuda()
            z = x + y
            print(f"Success! Result: {z.cpu().numpy()}")
        except Exception as e:
            print(f"Error during tensor operation: {e}")
    else:
        print("\nWARNING: CUDA is not available. The code will run on CPU, which is very slow for training.")
        print("Please ensure you have installed the CUDA-enabled version of PyTorch.")

if __name__ == "__main__":
    check_cuda()
