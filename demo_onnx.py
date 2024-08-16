#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse
from typing import Any, Tuple, List

import cv2
import numpy as np
import onnxruntime  # type: ignore


def run_inference(
    onnx_session: onnxruntime.InferenceSession,
    image: np.ndarray,
) -> int:
    input_detail = onnx_session.get_inputs()[0]
    input_name: str = input_detail.name
    input_shape: Tuple[int, int] = input_detail.shape[1:3]

    # Pre process: Resize, Normalize, Expand Dims, float32 cast
    input_image: np.ndarray = cv2.resize(
        image,
        dsize=(input_shape[1], input_shape[0]),
    )
    input_image = input_image / 255.0
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')

    # Inference
    result: Any = onnx_session.run(None, {input_name: input_image})

    # Post process: squeeze, MinMax Scale, uint8 cast
    result = np.squeeze(result)
    class_id: int = int(np.argmax(np.squeeze(result)))

    return class_id


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str, default='sample.jpg')
    parser.add_argument(
        "--model",
        type=str,
        default='model/nsfw_mobilenet2_224x224.onnx',
    )

    args = parser.parse_args()
    model_path: str = args.model
    image_path: str = args.image

    # Load model
    onnx_session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
        model_path,
        providers=[
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    )
    # Initial inference
    dummy_image: np.ndarray = np.zeros((256, 256, 3), dtype=np.uint8)
    _ = run_inference(
        onnx_session,
        dummy_image,
    )

    # Read image
    image: np.ndarray = cv2.imread(image_path)
    debug_image: np.ndarray = copy.deepcopy(image)

    start_time: float = time.time()

    # Inference execution
    class_id: int = run_inference(
        onnx_session,
        image,
    )

    elapsed_time: float = time.time() - start_time

    class_name: List[str] = [
        'drawings',
        'hentai',
        'neutral',
        'porn',
        'sexy',
    ]

    # Inference elapsed time
    cv2.putText(
        debug_image,
        "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(debug_image, "Class : " + class_name[class_id], (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

    # Display
    cv2.imshow('nsfw_model demo', debug_image)
    _ = cv2.waitKey(-1)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
