// Instantiate the service globally or inside your State controller class
final PlantClassifierService _classifierService = PlantClassifierService();

@override
void initState() {
  super.initState();
  // Fire off async system compilation tasks during controller creation
  _classifierService.initializeModel();
}

// Inside your picture-taking callback or image picker function:
Future<void> onImageSelected(File selectedFile) async {
  try {
    setState(() => _isAnalyzing = true);
    
    // Pass the image directly to the processing service
    LeafClassificationResult evaluation = await _classifierService.classifyLeafImage(selectedFile);
    
    // Render the final string outputs directly on screen via your state variables
    setState(() {
      _displayUiLabelText = "Diagnosis: ${evaluation.className}";
      _displayUiConfidenceText = "Confidence Score: ${(evaluation.confidence * 100).toStringAsFixed(1)}%";
      _isAnalyzing = false;
    });
  } catch (e) {
    print("UI Execution Error: $e");
    setState(() => _isAnalyzing = false);
  }
}

@override
void dispose() {
  _classifierService.close(); // Clean up system memory allocations safely
  super.dispose();
}