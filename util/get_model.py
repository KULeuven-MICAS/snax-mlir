#!/usr/bin/env python3
# Basic script to get a model
import argparse
from pathlib import Path

import requests


def get_model(model: str, output_file: Path) -> None:
    base_url = "https://github.com/mlcommons/tiny/raw/v1.0/benchmark/training"
    match model:
        case "ad01_int8":
            model_url = "/anomaly_detection/trained_models/ad01_int8.tflite"
        case "kws_ref_model":
            model_url = "/keyword_spotting/trained_models/kws_ref_model.tflite"
        case "pretrainedResnet_quant":
            model_url = (
                "/image_classification/trained_models/pretrainedResnet_quant.tflite"
            )
        case "vww_96_int8":
            model_url = "/visual_wake_words/trained_models/vww_96_int8.tflite"
        case _:
            raise Exception("No valid model selected")
    url = base_url + model_url
    # Download in a streaming fashion
    response = requests.get(url, stream=True)
    with open(output_file, mode="wb") as file:
        print(f"Downloading model '{model}' from '{url}'")
        for chunk in response.iter_content():
            file.write(chunk)
        print(f"Saved model to file '{output_file}'")


def main():
    parser = argparse.ArgumentParser(
        description="Download quantized MLPerf Tiny models"
    )
    parser.add_argument(
        "network",
        choices=[
            "anomaly_detection",
            "keyword_spotting",
            "image_classification",
            "visual_wake_words",
        ],
        help="Choose a neural network",
    )
    parser.add_argument("output_file", help="Output file")
    # Parse the command line arguments
    args = parser.parse_args()
    # Access the parsed arguments
    get_model(args.network, Path(args.output_file))


if __name__ == "__main__":
    main()
