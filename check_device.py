# import torch
import sys

def main():
    print(sys.executable)
    print(sys.version)

    # print("Torch version:", torch.__version__)
    # print("CUDA available:", torch.cuda.is_available())
    # print("CUDA version:", torch.version.cuda)
    # print("GPU count:", torch.cuda.device_count())

    # if torch.cuda.is_available():
    #     print("GPU name:", torch.cuda.get_device_name(0))
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(f'Device: {device}')
    

if __name__ == "__main__":
    main()