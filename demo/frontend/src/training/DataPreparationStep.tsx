import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  Upload,
  FileCheck,
  Split,
  AlertCircle,
  CheckCircle2,
  Loader2,
  FolderOutput,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Timeline, TimelineItem } from "@/components/ui/timeline";
import { apiClient } from "@/lib/api-client";

interface DataPreparationStepProps {
  onComplete: (data: any) => void;
  onBack: () => void;
  canGoBack: boolean;
}

type SubStep = "upload" | "convert" | "validate" | "split" | "complete";

interface DataPreparationState {
  currentSubStep: SubStep;
  sam2Path: string;
  outputDir: string;
  targetFormat: "llava" | "huggingface";
  uploadedFile: File | null;
  convertResult: any;
  validationResult: any;
  splitResult: any;
}

const STORAGE_KEY = "data-preparation-state";

export function DataPreparationStep({
  onComplete,
  onBack,
  canGoBack,
}: DataPreparationStepProps) {
  const [currentSubStep, setCurrentSubStep] = useState<SubStep>("upload");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Form state
  const [sam2Path, setSam2Path] = useState("");
  const [outputDir, setOutputDir] = useState("/app/output/training_data");
  const [targetFormat, setTargetFormat] = useState<"llava" | "huggingface">("llava");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);

  // Results state
  const [convertResult, setConvertResult] = useState<any>(null);
  const [validationResult, setValidationResult] = useState<any>(null);
  const [splitResult, setSplitResult] = useState<any>(null);

  // Load persisted state on mount
  useEffect(() => {
    const savedState = localStorage.getItem(STORAGE_KEY);
    if (savedState) {
      try {
        const state: DataPreparationState = JSON.parse(savedState);
        setCurrentSubStep(state.currentSubStep);
        setSam2Path(state.sam2Path);
        setOutputDir(state.outputDir);
        setTargetFormat(state.targetFormat);
        setConvertResult(state.convertResult);
        setValidationResult(state.validationResult);
        setSplitResult(state.splitResult);
        // Note: uploadedFile cannot be persisted to localStorage
      } catch (e) {
        console.error("Failed to load saved state:", e);
      }
    }
  }, []);

  // Persist state whenever it changes
  useEffect(() => {
    const state = {
      currentSubStep,
      sam2Path,
      outputDir,
      targetFormat,
      convertResult,
      validationResult,
      splitResult,
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  }, [currentSubStep, sam2Path, outputDir, targetFormat, convertResult, validationResult, splitResult]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.name.endsWith('.zip')) {
        setError('Please select a ZIP file');
        return;
      }
      setUploadedFile(file);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!uploadedFile) {
      setError('Please select a file first');
      return;
    }

    setUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', uploadedFile);

      const response = await fetch('/api/training/data/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const result = await response.json();
      setSam2Path(result.file_path);
      setError(null);

      // Auto-execute conversion after successful upload
      setTimeout(() => {
        handleConvert(result.file_path);
      }, 500);
    } catch (err: any) {
      setError(err.message || 'Upload failed');
      setUploading(false);
    }
  };

  const handleConvert = async (filePath?: string) => {
    const pathToUse = filePath || sam2Path;
    if (!pathToUse) {
      setError('Please upload a file first');
      return;
    }

    setLoading(true);
    setCurrentSubStep("convert");
    setError(null);

    try {
      const result = await apiClient.convertData({
        sam2_zip_path: pathToUse,
        output_dir: outputDir,
        target_format: targetFormat,
      });

      setConvertResult(result);
      setUploading(false);

      // Auto-execute validation after successful conversion
      setTimeout(() => {
        handleValidate(result);
      }, 500);
    } catch (err: any) {
      setError(err.message || "Conversion failed");
      setLoading(false);
      setUploading(false);
    }
  };

  const handleValidate = async (conversionResult?: any) => {
    const resultToUse = conversionResult || convertResult;

    setLoading(true);
    setCurrentSubStep("validate");
    setError(null);

    try {
      const dataPath = resultToUse?.output_dir;

      if (!dataPath) {
        throw new Error("No conversion result found. Please convert data first.");
      }

      const result = await apiClient.validateData({
        data_path: dataPath,
        format_type: targetFormat,
      });

      setValidationResult(result);

      // Auto-execute split after successful validation
      setTimeout(() => {
        handleSplit(resultToUse);
      }, 500);
    } catch (err: any) {
      setError(err.message || "Validation failed");
      setLoading(false);
    }
  };

  const handleSplit = async (conversionResult?: any) => {
    const resultToUse = conversionResult || convertResult;

    setLoading(true);
    setCurrentSubStep("split");
    setError(null);

    try {
      const dataPath = resultToUse?.output_dir;

      if (!dataPath) {
        throw new Error("No conversion result found. Please convert data first.");
      }

      const result = await apiClient.splitData({
        data_path: dataPath,
        output_dir: `${outputDir}/splits`,
        strategy: "stratified",
        train_ratio: 0.7,
        val_ratio: 0.2,
        test_ratio: 0.1,
        random_seed: 42,
      });

      setSplitResult(result);
      setCurrentSubStep("complete");
      setLoading(false);
    } catch (err: any) {
      setError(err.message || "Splitting failed");
      setLoading(false);
    }
  };

  const handleComplete = () => {
    // Clear persisted state on completion
    localStorage.removeItem(STORAGE_KEY);

    onComplete({
      sam2Path,
      outputDir,
      targetFormat,
      convertResult,
      validationResult,
      splitResult,
    });
  };

  const handleReset = () => {
    if (window.confirm("Are you sure you want to reset? All progress will be lost.")) {
      localStorage.removeItem(STORAGE_KEY);
      setCurrentSubStep("upload");
      setSam2Path("");
      setOutputDir("/app/output/training_data");
      setTargetFormat("llava");
      setUploadedFile(null);
      setConvertResult(null);
      setValidationResult(null);
      setSplitResult(null);
      setError(null);
    }
  };

  // Generate timeline items based on current state
  const getTimelineItems = (): TimelineItem[] => {
    const items: TimelineItem[] = [
      {
        id: "upload",
        title: "Upload Data",
        description: uploadedFile
          ? `File: ${uploadedFile.name} (${(uploadedFile.size / (1024 * 1024)).toFixed(2)} MB)`
          : "Upload SAM2 export ZIP file",
        status: sam2Path
          ? "completed"
          : (currentSubStep === "upload" || uploading)
            ? "active"
            : "pending",
        icon: <Upload className="h-3 w-3" />,
      },
      {
        id: "convert",
        title: "Convert Dataset",
        description: convertResult
          ? `Converted ${convertResult.num_samples} samples to ${targetFormat} format`
          : `Convert to ${targetFormat} format`,
        status: convertResult
          ? "completed"
          : currentSubStep === "convert"
            ? "active"
            : sam2Path
              ? "pending"
              : "default",
        icon: <FolderOutput className="h-3 w-3" />,
      },
      {
        id: "validate",
        title: "Validate Quality",
        description: validationResult
          ? `${validationResult.num_errors} errors, ${validationResult.num_warnings} warnings`
          : "Check dataset quality and balance",
        status: validationResult
          ? validationResult.passed ? "completed" : "error"
          : currentSubStep === "validate"
            ? "active"
            : convertResult
              ? "pending"
              : "default",
        icon: <FileCheck className="h-3 w-3" />,
      },
      {
        id: "split",
        title: "Split Dataset",
        description: splitResult
          ? `Train: ${splitResult.train_samples}, Val: ${splitResult.val_samples}, Test: ${splitResult.test_samples}`
          : "Divide into train/val/test sets (70/20/10)",
        status: splitResult
          ? "completed"
          : currentSubStep === "split"
            ? "active"
            : validationResult
              ? "pending"
              : "default",
        icon: <Split className="h-3 w-3" />,
      },
    ];

    return items;
  };

  return (
    <div className="space-y-6 p-6" style={{ backgroundColor: '#f9fafb', minHeight: '100vh' }}>
      {/* Help Banner */}
      <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
        <div className="flex items-start gap-3">
          <AlertCircle className="h-5 w-5 text-blue-500 mt-0.5 flex-shrink-0" />
          <div className="text-sm text-blue-800 flex-1">
            <p className="font-semibold mb-1">Data Preparation Pipeline</p>
            <p>Upload your SAM2 export file and the system will automatically process it through all steps: conversion, validation, and splitting.</p>
          </div>
        </div>
      </div>

      {/* Upload Settings Card - Always visible */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <Card className="border-2 border-indigo-200 bg-white">
          <CardHeader>
            <CardTitle className="text-gray-900">Upload Settings</CardTitle>
            <CardDescription className="text-gray-600">
              Select file and configure output settings
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Show saved configuration summary when not in upload state */}
            {currentSubStep !== "upload" && sam2Path && (
              <div className="p-4 bg-blue-50 border-2 border-blue-200 rounded-lg space-y-2">
                <div className="flex items-center gap-2 text-blue-900 font-medium">
                  <CheckCircle2 className="h-5 w-5 text-blue-600" />
                  <span>Configuration Saved</span>
                </div>
                <div className="text-sm text-blue-800 space-y-1 ml-7">
                  <p><strong>File:</strong> {sam2Path.split('/').pop()}</p>
                  <p><strong>Output:</strong> {outputDir}</p>
                  <p><strong>Format:</strong> {targetFormat.toUpperCase()}</p>
                </div>
              </div>
            )}

            {/* File Upload - Show when in upload state */}
            {currentSubStep === "upload" && (
              <div>
                <label className="block text-sm font-medium text-gray-900 mb-2">
                  Select SAM2 Export File (ZIP)
                </label>
                <div className="flex gap-3">
                  <input
                    type="file"
                    accept=".zip"
                    onChange={handleFileSelect}
                    disabled={uploading || loading}
                    className="flex-1 px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-all bg-white text-gray-900 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                  />
                </div>
                {uploadedFile && (
                  <p className="mt-2 text-sm text-gray-600">
                    Selected: {uploadedFile.name} ({(uploadedFile.size / (1024 * 1024)).toFixed(2)} MB)
                  </p>
                )}
                {sam2Path && !uploadedFile && (
                  <div className="mt-2 p-3 bg-green-50 border border-green-200 rounded">
                    <p className="text-sm text-green-800">
                      ✓ File uploaded: {sam2Path.split('/').pop()}
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Output Directory - Show when in upload state */}
            {currentSubStep === "upload" && (
              <div>
                <label className="block text-sm font-medium text-gray-900 mb-2">
                  Output Directory (Server Path)
                </label>
                <input
                  type="text"
                  value={outputDir}
                  onChange={(e) => setOutputDir(e.target.value)}
                  placeholder="/app/output/training_data"
                  className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-all bg-white text-gray-900"
                  disabled={uploading || loading}
                />
                <p className="mt-1 text-xs text-gray-500">Where processed data will be saved on the server</p>
              </div>
            )}

            {/* Target Format - Show when in upload state */}
            {currentSubStep === "upload" && (
              <div>
                <label className="block text-sm font-medium text-gray-900 mb-2">
                  Target Format
                </label>
                <div className="flex gap-3">
                  <button
                    onClick={() => setTargetFormat("llava")}
                    disabled={uploading || loading}
                    className={`flex-1 px-4 py-3 rounded-lg font-medium transition-all ${
                      targetFormat === "llava"
                        ? "bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg"
                        : "border-2 border-gray-200 hover:border-blue-300 text-gray-700"
                    }`}
                  >
                    LLaVA Format
                  </button>
                  <button
                    onClick={() => setTargetFormat("huggingface")}
                    disabled={uploading || loading}
                    className={`flex-1 px-4 py-3 rounded-lg font-medium transition-all ${
                      targetFormat === "huggingface"
                        ? "bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg"
                        : "border-2 border-gray-200 hover:border-blue-300 text-gray-700"
                    }`}
                  >
                    HuggingFace Format
                  </button>
                </div>
              </div>
            )}

            {/* Start Button */}
            {currentSubStep === "upload" && (
              <Button
                onClick={handleUpload}
                disabled={!uploadedFile || uploading || loading}
                className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white py-3 text-base font-medium"
              >
                {uploading || loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Processing Pipeline...
                  </>
                ) : (
                  <>
                    <Upload className="mr-2 h-4 w-4" />
                    Start Pipeline
                  </>
                )}
              </Button>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Progress Timeline */}
      <Card className="border-2 border-blue-200 bg-white">
        <CardHeader>
          <CardTitle className="text-gray-900">Processing Pipeline</CardTitle>
          <CardDescription className="text-gray-600">
            {currentSubStep === "complete"
              ? "All steps completed! Ready to continue."
              : loading || uploading
                ? "Processing your data..."
                : currentSubStep === "upload"
                  ? "Ready to start - select a file and click Start Pipeline"
                  : "Processing in progress..."}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Timeline
            items={getTimelineItems()}
            variant="default"
            showTimestamps={false}
          />
        </CardContent>
      </Card>

      {/* Results Card - shown during and after processing */}
      {currentSubStep !== "upload" && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Card className="border-2 border-green-200 bg-white">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-gray-900">Results</CardTitle>
                  <CardDescription className="text-gray-600">
                    Pipeline execution results
                  </CardDescription>
                </div>
                {currentSubStep === "complete" && (
                  <Badge variant="success" className="text-base px-4 py-2">
                    <CheckCircle2 className="h-4 w-4 mr-2" />
                    All Steps Complete
                  </Badge>
                )}
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Conversion Results */}
              {convertResult && (
                <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                  <div className="flex items-start gap-3">
                    <CheckCircle2 className="h-5 w-5 text-green-600 mt-0.5" />
                    <div className="flex-1">
                      <p className="font-medium text-green-900">Conversion Complete</p>
                      <p className="text-sm text-green-800 mt-1">
                        Converted {convertResult.num_samples} samples to {targetFormat} format
                      </p>
                      <p className="text-xs text-green-700 mt-1">
                        Output: {convertResult.output_dir}
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Validation Results */}
              {validationResult && (
                <div className={`p-4 border rounded-lg ${
                  validationResult.passed
                    ? "bg-green-50 border-green-200"
                    : "bg-yellow-50 border-yellow-200"
                }`}>
                  <div className="flex items-start gap-3">
                    {validationResult.passed ? (
                      <CheckCircle2 className="h-5 w-5 text-green-600 mt-0.5" />
                    ) : (
                      <AlertCircle className="h-5 w-5 text-yellow-600 mt-0.5" />
                    )}
                    <div className="flex-1">
                      <p className={`font-medium ${
                        validationResult.passed ? "text-green-900" : "text-yellow-900"
                      }`}>
                        Validation {validationResult.passed ? "Passed" : "Completed with Warnings"}
                      </p>
                      <div className="mt-2 space-y-1">
                        <p className="text-sm">
                          <span className="font-medium">Errors:</span>{" "}
                          <Badge variant={validationResult.num_errors > 0 ? "destructive" : "success"}>
                            {validationResult.num_errors}
                          </Badge>
                        </p>
                        <p className="text-sm">
                          <span className="font-medium">Warnings:</span>{" "}
                          <Badge variant={validationResult.num_warnings > 0 ? "warning" : "success"}>
                            {validationResult.num_warnings}
                          </Badge>
                        </p>
                      </div>
                      {validationResult.recommendations?.length > 0 && (
                        <div className="mt-3 p-2 bg-white rounded">
                          <p className="text-xs font-medium mb-1">Recommendations:</p>
                          <ul className="text-xs space-y-1">
                            {validationResult.recommendations.map((rec: string, i: number) => (
                              <li key={i}>• {rec}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Split Results */}
              {splitResult && (
                <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                  <div className="flex items-start gap-3">
                    <CheckCircle2 className="h-5 w-5 text-green-600 mt-0.5" />
                    <div className="flex-1">
                      <p className="font-medium text-green-900">Dataset Split Complete</p>
                      <div className="mt-2 grid grid-cols-3 gap-3">
                        <div className="p-2 bg-white rounded text-center">
                          <p className="text-xs text-gray-600">Training</p>
                          <p className="text-lg font-bold text-gray-900">{splitResult.train_samples}</p>
                        </div>
                        <div className="p-2 bg-white rounded text-center">
                          <p className="text-xs text-gray-600">Validation</p>
                          <p className="text-lg font-bold text-gray-900">{splitResult.val_samples}</p>
                        </div>
                        <div className="p-2 bg-white rounded text-center">
                          <p className="text-xs text-gray-600">Test</p>
                          <p className="text-lg font-bold text-gray-900">{splitResult.test_samples}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Processing Indicator */}
              {loading && currentSubStep !== "complete" && (
                <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <div className="flex items-center gap-3">
                    <Loader2 className="h-5 w-5 text-blue-600 animate-spin" />
                    <p className="text-sm text-blue-800 font-medium">
                      Processing step: {currentSubStep}...
                    </p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* Error Display */}
      {error && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="p-4 bg-red-50 border-2 border-red-200 rounded-lg flex items-start gap-3"
        >
          <AlertCircle className="h-5 w-5 text-red-600 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="font-medium text-red-900">Error</p>
            <p className="text-sm text-red-700">{error}</p>
          </div>
        </motion.div>
      )}

      {/* Navigation */}
      <div className="flex gap-3">
        {canGoBack && (
          <Button variant="outline" onClick={onBack} className="flex-1">
            Back
          </Button>
        )}
        {currentSubStep !== "upload" && currentSubStep !== "complete" && (
          <Button variant="outline" onClick={handleReset} className="flex-1">
            Reset Pipeline
          </Button>
        )}
        {currentSubStep === "complete" && (
          <Button
            onClick={handleComplete}
            className="flex-1 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700"
          >
            Continue to Training Config
          </Button>
        )}
      </div>
    </div>
  );
}
