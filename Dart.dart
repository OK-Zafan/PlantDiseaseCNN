import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

/// Structured container class representing the final disease evaluation.
class LeafClassificationResult {
  final String className;
  final double confidence;

  LeafClassificationResult({
    required this.className,
    required this.confidence,
  });

  @override
  String toString() => '$className (${(confidence * 100).toStringAsFixed(2)}%)';
}

class PlantClassifierService {
  Interpreter? _interpreter;
  List<String>? _labels;
  bool _isModelLoaded = false;

  // Constants mapping strictly to the model card specifications
  static const String _modelPath = 'assets/potato_model_mobilenetv2_v2.tflite';
  static const String _labelsPath = 'assets/class_names.json';
  static const int _inputSize = 224;

  bool get isModelLoaded => _isModelLoaded;

  /// Asynchronously allocates runtime memory and initializes the TFLite execution graph.
  Future<void> initializeModel() async {
    if (_isModelLoaded) return;

    try {
      // 1. Configure deployment runtime options for edge hardware acceleration
      final options = InterpreterOptions();
      if (Platform.isAndroid) {
        options.addDelegate(XNNPackDelegate()); // Speeds up inference via CPU multithreading
      } else if (Platform.isIOS) {
        options.addDelegate(GpuDelegate2()); // Hardware Metal acceleration on iOS devices
      }

      // 2. Instantiate the isolated execution interpreter graph
      _interpreter = await Interpreter.fromAsset(_modelPath, options: options);

      // 3. Load the class labels array mapping configurations
      final labelsJsonContent = await rootBundle.loadString(_labelsPath);
      final List<dynamic> rawLabelsList = json.decode(labelsJsonContent);
      _labels = rawLabelsList.map((item) => item.toString()).toList();

      _isModelLoaded = true;
      print('TFLite Model Engine initialized successfully.');
      print('Model Input Tensor Spec: ${_interpreter!.getInputTensors().first.shape}');
      print('Model Output Tensor Spec: ${_interpreter!.getOutputTensors().first.shape}');
    } catch (e) {
      print('CRITICAL: Failed to load or initialize TFLite model components: $e');
      _isModelLoaded = false;
    }
  }

  /// Takes a local image file, prepares the pixel buffers, runs inference, and parses the result.
  Future<LeafClassificationResult> classifyLeafImage(File imageFile) async {
    if (!_isModelLoaded || _interpreter == null || _labels == null) {
      throw StateError('Classifier execution attempted before model initialization was complete.');
    }

    // 1. Read byte stream and decode via image processing library
    final Uint8List imageBytes = await imageFile.readAsBytes();
    final img.Image? decodedImage = img.decodeImage(imageBytes);

    if (decodedImage == null) {
      throw ArgumentError('Unable to parse or decode file format into valid bitmap imagery.');
    }

    // 2. Downsample and reshape input matrix dimensions to match the target 224x224 shape via Bilinear interpolation
    final img.Image resizedImage = img.copyResize(
      decodedImage,
      width: _inputSize,
      height: _inputSize,
      interpolation: img.Interpolation.bilinear,
    );

    // 3. Convert image pixels into a normalized Float32 byte buffer using the fixed model formula
    // Normalization logic: V_normalized = (Raw_Byte_Value / 127.5) - 1.0 (Maps [0, 255] directly to [-1.0, 1.0])
    final Float32List inputBuffer = Float32List(_inputSize * _inputSize * 3);
    int bufferIndex = 0;

    for (int y = 0; y < _inputSize; y++) {
      for (int x = 0; x < _inputSize; x++) {
        final pixel = resizedImage.getPixel(x, y);

        // Extract native channel integers safely regardless of individual bitmap configuration headers
        final double r = pixel.r.toDouble();
        final double g = pixel.g.toDouble();
        final double b = pixel.b.toDouble();

        // Sequential entry insertion matching standard RGB channel ordering constraints
        inputBuffer[bufferIndex++] = (r / 127.5) - 1.0;
        inputBuffer[bufferIndex++] = (g / 127.5) - 1.0;
        inputBuffer[bufferIndex++] = (b / 127.5) - 1.0;
      }
    }

    // 4. Shape the 1D buffer into a multi-dimensional matrix shape array: [1, 224, 224, 3]
    final inputTensor = inputBuffer.reshape([1, _inputSize, _inputSize, 3]);

    // 5. Pre-allocate target allocation output matrix container shape: [1, 3]
    final outputTensor = List<double>.filled(3, 0.0).reshape([1, 3]);

    // 6. Execute model inference inside the compiled engine
    _interpreter!.run(inputTensor, outputTensor);

    // 7. Parse the multi-dimensional output array safely
    final List<double> softmaxProbabilities = List<double>.from(outputTensor[0]);
    print('Raw Inference Output Softmax Probabilities: $softmaxProbabilities');

    // 8. Find the index with the maximum confidence level (ArgMax extraction)
    double maxConfidence = -1.0;
    int bestMatchedClassIndex = 0;

    for (int i = 0; i < softmaxProbabilities.length; i++) {
      if (softmaxProbabilities[i] > maxConfidence) {
        maxConfidence = softmaxProbabilities[i];
        bestMatchedClassIndex = i;
      }
    }

    // 9. Format output string presentation mapping parameters cleanly
    final String rawClassName = _labels![bestMatchedClassIndex];
    final String readableClassName = rawClassName.replaceAll('___', ' ').replaceAll('_', ' ');

    return LeafClassificationResult(
      className: readableClassName,
      confidence: maxConfidence,
    );
  }

  /// Releases model memory bindings when the application lifecycle ends.
  void close() {
    _interpreter?.close();
    _isModelLoaded = false;
    print('TFLite Model Engine resources freed successfully.');
  }
}