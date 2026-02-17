# from em_inference_app.utils.inference_utils import predict

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) < 2:
#         print("Usage: predict-nuclei path/to/image.tif")
#         sys.exit(1)
#     predict(sys.argv[1])



import argparse
from em_inference_app.utils.inference_utils import predict

def main():
    parser = argparse.ArgumentParser(description="Run nuclei segmentation inference")
    parser.add_argument("image_path", type=str)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    predict(args.image_path, threshold=args.threshold)

if __name__ == "__main__":
    main()