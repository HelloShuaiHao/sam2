import { useState } from "react";
import { motion } from "framer-motion";
import {
  Upload,
  FileCheck,
  Split,
  AlertCircle,
  CheckCircle2,
  Loader2,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { apiClient } from "@/lib/api-client";

interface DataPreparationStepProps {
  onComplete: (data: any) => void;
  onBack: () => void;
  canGoBack: boolean;
}

type SubStep = "upload" | "validate" | "split" | "complete";

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
    } catch (err: any) {
      setError(err.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleConvert = async () => {
    if (!sam2Path) {
      setError('Please upload a file first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await apiClient.convertData({
        sam2_zip_path: sam2Path,
        output_dir: outputDir,
        target_format: targetFormat,
      });

      setConvertResult(result);
      setCurrentSubStep("validate");
    } catch (err: any) {
      setError(err.message || "Conversion failed");
    } finally {
      setLoading(false);
    }
  };

  const handleValidate = async () => {
    setLoading(true);
    setError(null);

    try {
      const dataPath =
        targetFormat === "llava"
          ? `${outputDir}/llava_format.jsonl`
          : `${outputDir}/huggingface_dataset`;

      const result = await apiClient.validateData({
        data_path: dataPath,
        format_type: targetFormat,
      });

      setValidationResult(result);
      setCurrentSubStep("split");
    } catch (err: any) {
      setError(err.message || "Validation failed");
    } finally {
      setLoading(false);
    }
  };

  const handleSplit = async () => {
    setLoading(true);
    setError(null);

    try {
      const dataPath =
        targetFormat === "llava"
          ? `${outputDir}/llava_format.jsonl`
          : `${outputDir}/huggingface_dataset`;

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
    } catch (err: any) {
      setError(err.message || "Splitting failed");
    } finally {
      setLoading(false);
    }
  };

  const handleComplete = () => {
    onComplete({
      sam2Path,
      outputDir,
      targetFormat,
      convertResult,
      validationResult,
      splitResult,
    });
  };

  return (
    <div className="space-y-6 p-6" style={{ backgroundColor: '#f9fafb', minHeight: '100vh' }}>
      {/* Help Banner */}
      <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
        <div className="flex items-start gap-3">
          <AlertCircle className="h-5 w-5 text-blue-500 mt-0.5 flex-shrink-0" />
          <div className="text-sm text-blue-800">
            <p className="font-semibold mb-1">📁 File Paths Guide:</p>
            <ul className="list-disc list-inside space-y-1 ml-2">
              <li><strong>SAM2 Export Path:</strong> Server path to your exported ZIP file (e.g., <code className="bg-blue-100 px-1 rounded">/data/exports/my_video.zip</code>)</li>
              <li><strong>Output Directory:</strong> Server path where processed data will be saved (e.g., <code className="bg-blue-100 px-1 rounded">/data/training_data</code>)</li>
              <li><strong>Note:</strong> All paths are on the server running the Training API container</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Upload & Convert */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <Card className="border-2 border-blue-200 bg-white">
          <CardHeader>
            <div className="flex items-center gap-3">
              <div className="p-3 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl shadow-lg">
                <Upload className="h-6 w-6 text-white" />
              </div>
              <div className="flex-1">
                <CardTitle className="text-gray-900">1. Upload & Convert</CardTitle>
                <CardDescription className="text-gray-600">Convert SAM2 export to training format</CardDescription>
              </div>
              {convertResult && (
                <Badge variant="success">
                  <CheckCircle2 className="h-3 w-3 mr-1" />
                  Completed
                </Badge>
              )}
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* File Upload */}
            <div>
              <label className="block text-sm font-medium text-gray-900 mb-2">
                1. Select SAM2 Export File (ZIP)
              </label>
              <div className="flex gap-3">
                <input
                  type="file"
                  accept=".zip"
                  onChange={handleFileSelect}
                  disabled={currentSubStep !== "upload"}
                  className="flex-1 px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-all bg-white text-gray-900 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                />
                <Button
                  onClick={handleUpload}
                  disabled={!uploadedFile || uploading || currentSubStep !== "upload"}
                  className="px-6"
                >
                  {uploading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Uploading...
                    </>
                  ) : (
                    <>
                      <Upload className="mr-2 h-4 w-4" />
                      Upload
                    </>
                  )}
                </Button>
              </div>
              {uploadedFile && (
                <p className="mt-2 text-sm text-gray-600">
                  Selected: {uploadedFile.name} ({(uploadedFile.size / (1024 * 1024)).toFixed(2)} MB)
                </p>
              )}
              {sam2Path && (
                <div className="mt-2 p-3 bg-green-50 border border-green-200 rounded">
                  <p className="text-sm text-green-800">
                    ✓ File uploaded: {sam2Path}
                  </p>
                </div>
              )}
            </div>

            {/* Output Directory */}
            <div>
              <label className="block text-sm font-medium text-gray-900 mb-2">
                2. Output Directory (Server Path)
              </label>
              <input
                type="text"
                value={outputDir}
                onChange={(e) => setOutputDir(e.target.value)}
                placeholder="/app/output/training_data"
                className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-all bg-white text-gray-900"
                style={{ color: '#111827' }}
                disabled={currentSubStep !== "upload"}
              />
              <p className="mt-1 text-xs text-gray-500">This is where processed data will be saved on the server</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-900 mb-2">
                3. Target Format
              </label>
              <div className="flex gap-3">
                <button
                  onClick={() => setTargetFormat("llava")}
                  disabled={currentSubStep !== "upload"}
                  className={`flex-1 px-4 py-3 rounded-lg font-medium transition-all ${
                    targetFormat === "llava"
                      ? "bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg"
                      : "border-2 border-gray-200 hover:border-blue-300"
                  }`}
                >
                  LLaVA Format
                </button>
                <button
                  onClick={() => setTargetFormat("huggingface")}
                  disabled={currentSubStep !== "upload"}
                  className={`flex-1 px-4 py-3 rounded-lg font-medium transition-all ${
                    targetFormat === "huggingface"
                      ? "bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg"
                      : "border-2 border-gray-200 hover:border-blue-300"
                  }`}
                >
                  HuggingFace Format
                </button>
              </div>
            </div>

            {convertResult && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="p-4 bg-green-50 border border-green-200 rounded-lg"
              >
                <p className="text-sm text-green-800">
                  ✓ Converted {convertResult.num_samples} samples to {targetFormat} format
                </p>
              </motion.div>
            )}

            {error && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-sm text-red-800">{error}</p>
              </div>
            )}

            {currentSubStep === "upload" && (
              <Button
                onClick={handleConvert}
                disabled={!sam2Path || loading}
                className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white py-3"
              >
                {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                4. Convert Dataset
              </Button>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Validate */}
      {currentSubStep !== "upload" && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <Card className="border-2 border-purple-200">
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="p-3 bg-gradient-to-r from-purple-500 to-pink-600 rounded-xl shadow-lg">
                  <FileCheck className="h-6 w-6 text-white" />
                </div>
                <div className="flex-1">
                  <CardTitle>2. Validate Quality</CardTitle>
                  <CardDescription>Check dataset quality and balance</CardDescription>
                </div>
                {validationResult && (
                  <Badge variant={validationResult.passed ? "success" : "warning"}>
                    {validationResult.passed ? (
                      <>
                        <CheckCircle2 className="h-3 w-3 mr-1" />
                        Passed
                      </>
                    ) : (
                      <>
                        <AlertCircle className="h-3 w-3 mr-1" />
                        Warnings
                      </>
                    )}
                  </Badge>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {validationResult ? (
                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Errors:</span>
                    <Badge variant={validationResult.num_errors > 0 ? "destructive" : "success"}>
                      {validationResult.num_errors}
                    </Badge>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Warnings:</span>
                    <Badge variant={validationResult.num_warnings > 0 ? "warning" : "success"}>
                      {validationResult.num_warnings}
                    </Badge>
                  </div>
                  {validationResult.recommendations?.length > 0 && (
                    <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                      <p className="text-sm font-medium text-blue-900 mb-2">
                        Recommendations:
                      </p>
                      <ul className="text-xs text-blue-800 space-y-1">
                        {validationResult.recommendations.map((rec: string, i: number) => (
                          <li key={i}>• {rec}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ) : currentSubStep === "validate" ? (
                <Button onClick={handleValidate} disabled={loading} className="w-full">
                  {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                  Validate Dataset
                </Button>
              ) : null}
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* Split */}
      {currentSubStep !== "upload" && currentSubStep !== "validate" && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <Card className="border-2 border-green-200">
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="p-3 bg-gradient-to-r from-green-500 to-emerald-600 rounded-xl shadow-lg">
                  <Split className="h-6 w-6 text-white" />
                </div>
                <div className="flex-1">
                  <CardTitle>3. Split Dataset</CardTitle>
                  <CardDescription>Divide into train/val/test sets (70/20/10)</CardDescription>
                </div>
                {splitResult && (
                  <Badge variant="success">
                    <CheckCircle2 className="h-3 w-3 mr-1" />
                    Completed
                  </Badge>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {splitResult ? (
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Training samples:</span>
                    <Badge>{splitResult.train_samples}</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Validation samples:</span>
                    <Badge>{splitResult.val_samples}</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Test samples:</span>
                    <Badge>{splitResult.test_samples}</Badge>
                  </div>
                </div>
              ) : currentSubStep === "split" ? (
                <Button onClick={handleSplit} disabled={loading} className="w-full">
                  {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                  Split Dataset
                </Button>
              ) : null}
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
        {currentSubStep === "complete" && (
          <Button onClick={handleComplete} className="flex-1">
            Continue to Training Config
          </Button>
        )}
      </div>
    </div>
  );
}
