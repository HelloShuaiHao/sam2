import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  Download,
  Package,
  FileCheck,
  Loader2,
  CheckCircle2,
  ExternalLink,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { apiClient } from "@/lib/api-client";
import { formatBytes } from "@/lib/utils";

interface ExportStepProps {
  jobId?: string;
  onBack: () => void;
  canGoBack: boolean;
}

type ExportFormat = "huggingface" | "lora_adapter";

export function ExportStep({ jobId, onBack, canGoBack }: ExportStepProps) {
  const [exportFormat, setExportFormat] = useState<ExportFormat>("lora_adapter");
  const [loading, setLoading] = useState(false);
  const [exportResult, setExportResult] = useState<any>(null);
  const [downloadInfo, setDownloadInfo] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleExport = async () => {
    if (!jobId) return;

    setLoading(true);
    setError(null);

    try {
      const result = await apiClient.exportModel(jobId, {
        export_format: exportFormat,
        generate_model_card: true,
        merge_adapters: exportFormat === "huggingface",
      });

      setExportResult(result);

      // Get download info
      const dlInfo = await apiClient.getDownloadInfo(jobId);
      setDownloadInfo(dlInfo);
    } catch (err: any) {
      setError(err.message || "Export failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Export Format Selection */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <Card className="border-2 border-blue-200">
          <CardHeader>
            <div className="flex items-center gap-3">
              <div className="p-3 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl shadow-lg">
                <Package className="h-6 w-6 text-white" />
              </div>
              <div>
                <CardTitle>Export Format</CardTitle>
                <CardDescription>Choose how to package your model</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4">
              <motion.button
                onClick={() => setExportFormat("lora_adapter")}
                disabled={!!exportResult}
                className={`relative p-6 rounded-xl border-2 text-left transition-all ${
                  exportFormat === "lora_adapter"
                    ? "border-blue-500 bg-gradient-to-r from-blue-50 to-indigo-50 shadow-lg"
                    : "border-gray-200 hover:border-gray-300"
                }`}
                whileHover={{ scale: exportResult ? 1 : 1.02 }}
                whileTap={{ scale: exportResult ? 1 : 0.98 }}
              >
                <div className="absolute -top-3 -right-3">
                  <Badge variant="success">
                    <CheckCircle2 className="h-3 w-3 mr-1" />
                    Recommended
                  </Badge>
                </div>

                <h3 className="font-bold text-lg mb-2">LoRA Adapters Only</h3>
                <p className="text-sm text-gray-600 mb-3">
                  Export only the fine-tuned adapter weights (~10-50 MB). Requires base model to use.
                </p>
                <div className="flex gap-2">
                  <Badge variant="outline">Small Size</Badge>
                  <Badge variant="secondary">Fast Download</Badge>
                </div>
              </motion.button>

              <motion.button
                onClick={() => setExportFormat("huggingface")}
                disabled={!!exportResult}
                className={`p-6 rounded-xl border-2 text-left transition-all ${
                  exportFormat === "huggingface"
                    ? "border-blue-500 bg-gradient-to-r from-blue-50 to-indigo-50 shadow-lg"
                    : "border-gray-200 hover:border-gray-300"
                }`}
                whileHover={{ scale: exportResult ? 1 : 1.02 }}
                whileTap={{ scale: exportResult ? 1 : 0.98 }}
              >
                <h3 className="font-bold text-lg mb-2">Full Model (HuggingFace)</h3>
                <p className="text-sm text-gray-600 mb-3">
                  Export complete model with merged weights (~13-26 GB). Ready to use standalone.
                </p>
                <div className="flex gap-2">
                  <Badge variant="outline">Large Size</Badge>
                  <Badge variant="outline">Standalone</Badge>
                </div>
              </motion.button>
            </div>

            {!exportResult && (
              <Button
                onClick={handleExport}
                disabled={loading || !jobId}
                className="w-full"
              >
                {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                Export Model
              </Button>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Export Results */}
      {exportResult && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Card className="border-2 border-green-200 bg-gradient-to-r from-green-50 to-emerald-50">
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="p-3 bg-gradient-to-r from-green-500 to-emerald-600 rounded-xl shadow-lg">
                  <CheckCircle2 className="h-6 w-6 text-white" />
                </div>
                <div>
                  <CardTitle className="text-green-900">Export Completed!</CardTitle>
                  <CardDescription className="text-green-700">
                    Your model has been packaged and is ready for download
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <span className="text-sm text-gray-600">Export Format:</span>
                  <p className="font-semibold text-gray-900">
                    {exportResult.export_format === "lora_adapter"
                      ? "LoRA Adapters"
                      : "HuggingFace Full Model"}
                  </p>
                </div>
                <div>
                  <span className="text-sm text-gray-600">File Size:</span>
                  <p className="font-semibold text-gray-900">
                    {formatBytes(exportResult.file_size_mb * 1024 * 1024)}
                  </p>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Export Path:</span>
                  <p className="font-mono text-xs text-gray-700 truncate">
                    {exportResult.export_path}
                  </p>
                </div>
                {exportResult.model_card_path && (
                  <div>
                    <span className="text-sm text-gray-600">Model Card:</span>
                    <p className="font-mono text-xs text-gray-700 truncate">
                      {exportResult.model_card_path}
                    </p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* Download Card */}
      {downloadInfo && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Card className="border-2 border-purple-200">
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="p-3 bg-gradient-to-r from-purple-500 to-pink-600 rounded-xl shadow-lg">
                  <Download className="h-6 w-6 text-white" />
                </div>
                <div>
                  <CardTitle>Download Model</CardTitle>
                  <CardDescription>Get your trained model</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="p-4 bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg border border-purple-200">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <p className="font-semibold text-purple-900">{downloadInfo.filename}</p>
                    <p className="text-sm text-purple-700">
                      {formatBytes(downloadInfo.file_size_mb * 1024 * 1024)}
                    </p>
                  </div>
                  <FileCheck className="h-8 w-8 text-purple-600" />
                </div>

                <a
                  href={`http://localhost:8000${downloadInfo.download_url}`}
                  download={downloadInfo.filename}
                  className="block w-full"
                >
                  <Button className="w-full" size="lg">
                    <Download className="mr-2 h-5 w-5" />
                    Download Model Package
                  </Button>
                </a>
              </div>

              {downloadInfo.expires_at && (
                <p className="text-xs text-gray-500 text-center">
                  Download link expires on{" "}
                  {new Date(downloadInfo.expires_at).toLocaleDateString()}
                </p>
              )}
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* Usage Instructions */}
      {exportResult && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <Card>
            <CardHeader>
              <CardTitle>Usage Instructions</CardTitle>
              <CardDescription>How to use your exported model</CardDescription>
            </CardHeader>
            <CardContent>
              {exportFormat === "lora_adapter" ? (
                <div className="space-y-3">
                  <p className="text-sm text-gray-700">
                    Load your LoRA adapters with the base model:
                  </p>
                  <pre className="p-4 bg-gray-900 text-gray-100 rounded-lg text-xs overflow-x-auto">
{`from peft import PeftModel
from transformers import AutoModel

# Load base model
base_model = AutoModel.from_pretrained("${jobId ? 'your-base-model' : ''}")

# Load LoRA adapters
model = PeftModel.from_pretrained(
    base_model,
    "path/to/lora_adapters"
)

# Use model for inference
model.eval()`}
                  </pre>
                </div>
              ) : (
                <div className="space-y-3">
                  <p className="text-sm text-gray-700">
                    Load your merged model directly:
                  </p>
                  <pre className="p-4 bg-gray-900 text-gray-100 rounded-lg text-xs overflow-x-auto">
{`from transformers import AutoModel

# Load model with merged weights
model = AutoModel.from_pretrained(
    "path/to/huggingface_model"
)

# Use model for inference
model.eval()`}
                  </pre>
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
          className="p-4 bg-red-50 border-2 border-red-200 rounded-lg"
        >
          <p className="text-sm text-red-700">{error}</p>
        </motion.div>
      )}

      {/* Navigation */}
      <div className="flex gap-3">
        {canGoBack && !exportResult && (
          <Button variant="outline" onClick={onBack} className="flex-1">
            Back to Training
          </Button>
        )}
        {exportResult && (
          <Button
            variant="outline"
            onClick={() => window.location.reload()}
            className="flex-1"
          >
            <ExternalLink className="mr-2 h-4 w-4" />
            Start New Training
          </Button>
        )}
      </div>
    </div>
  );
}
