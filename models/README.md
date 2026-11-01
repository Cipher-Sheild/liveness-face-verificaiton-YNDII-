# Pretrained Models Directory

This directory stores the pretrained model weights for the `Silent-Face-Anti-Spoofing` model, as well as the OpenCV DNN face detection models.

## OpenCV Face Detection Models
The face detector relies on `res10_300x300_ssd_iter_140000.caffemodel` and its configuration file `deploy.prototxt`.
A script is provided in `src/face_detector.py` to automatically download these models if they are not found.

## Silent-Face-Anti-Spoofing Model Weights
We use the pretrained ONNX models from [minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing).

### Automatic Download
The `4_0_0_80x80_MiniFASNetV1SE.onnx` model weights are now **automatically downloaded** on the first run by `src/anti_spoof_model.py` from a reliable open-source fork (`QingHeYang/Silent-Face-Anti-Spoofing-onnx`).

### Manual Download (Fallback)
If you operate in an offline environment or the automatic download fails, you can manually download the `4_0_0_80x80_MiniFASNetV1SE.onnx` file from:
`https://raw.githubusercontent.com/QingHeYang/Silent-Face-Anti-Spoofing-onnx/master/onnx/4_0_0_80x80_MiniFASNetV1SE.onnx`

Place the downloaded file directly into this `models/` directory.

**Note**: Since we strictly depend on the ONNX version to avoid heavy PyTorch dependencies, it is imperative to secure this `.onnx` file for the model wrapper to function.
