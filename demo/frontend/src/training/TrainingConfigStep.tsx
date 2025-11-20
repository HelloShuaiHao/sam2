import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  Brain,
  Zap,
  Cpu,
  AlertCircle,
  Info,
  Sparkles,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

interface TrainingConfigStepProps {
  dataConfig: any;
  onComplete: (config: any) => void;
  onBack: () => void;
  canGoBack: boolean;
}

const modelPresets = [
  {
    id: "llava-7b-qlora",
    name: "LLaVA 7B (QLoRA)",
    description: "Best for 8GB GPU - Ultra memory efficient",
    modelName: "liuhaotian/llava-v1.5-7b",
    vramEstimate: "~6GB",
    recommended: true,
    config: {
      use_lora: true,
      use_qlora: true,
      lora_rank: 8,
      lora_alpha: 16,
      batch_size: 1,
      gradient_accumulation_steps: 8,
      max_length: 1024,
    },
  },
  {
    id: "llava-7b-lora",
    name: "LLaVA 7B (LoRA)",
    description: "Requires 16GB+ GPU",
    modelName: "liuhaotian/llava-v1.5-7b",
    vramEstimate: "~12GB",
    recommended: false,
    config: {
      use_lora: true,
      use_qlora: false,
      lora_rank: 16,
      lora_alpha: 32,
      batch_size: 2,
      gradient_accumulation_steps: 4,
      max_length: 2048,
    },
  },
  {
    id: "qwen-vl",
    name: "Qwen-VL (QLoRA)",
    description: "Alternative vision model - 8GB compatible",
    modelName: "Qwen/Qwen-VL",
    vramEstimate: "~7GB",
    recommended: false,
    config: {
      use_lora: true,
      use_qlora: true,
      lora_rank: 8,
      lora_alpha: 16,
      batch_size: 1,
      gradient_accumulation_steps: 8,
      max_length: 1024,
    },
  },
];

export function TrainingConfigStep({
  dataConfig,
  onComplete,
  onBack,
  canGoBack,
}: TrainingConfigStepProps) {
  const [selectedPreset, setSelectedPreset] = useState(modelPresets[0]);
  const [customConfig, setCustomConfig] = useState({
    num_epochs: 3,
    learning_rate: 0.0002,
    warmup_ratio: 0.03,
    save_steps: 100,
    eval_steps: 100,
  });

  const handleComplete = () => {
    const fullConfig = {
      model_name: selectedPreset.modelName,
      ...selectedPreset.config,
      ...customConfig,
      train_data_path: dataConfig.splitResult.train_path,
      val_data_path: dataConfig.splitResult.val_path,
      output_dir: `${dataConfig.outputDir}/checkpoints`,
      fp16: true,
      bf16: false,
      device: "cuda",
      logging_steps: 10,
      save_total_limit: 3,
    };

    onComplete(fullConfig);
  };

  return (
    <div className="space-y-6">
      {/* Model Selection */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <Card className="border-2 border-blue-200">
          <CardHeader>
            <div className="flex items-center gap-3">
              <div className="p-3 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl shadow-lg">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <div>
                <CardTitle>Model Selection</CardTitle>
                <CardDescription>Choose a pre-configured model preset</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4">
              {modelPresets.map((preset) => (
                <motion.button
                  key={preset.id}
                  onClick={() => setSelectedPreset(preset)}
                  className={`relative p-6 rounded-xl border-2 text-left transition-all ${
                    selectedPreset.id === preset.id
                      ? "border-blue-500 bg-gradient-to-r from-blue-50 to-indigo-50 shadow-lg"
                      : "border-gray-200 hover:border-gray-300 hover:shadow-md"
                  }`}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  {preset.recommended && (
                    <div className="absolute -top-3 -right-3">
                      <Badge variant="success" className="shadow-lg">
                        <Sparkles className="h-3 w-3 mr-1" />
                        Recommended for 8GB
                      </Badge>
                    </div>
                  )}

                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1">
                      <h3 className="font-bold text-lg mb-1">{preset.name}</h3>
                      <p className="text-sm text-gray-600 mb-3">{preset.description}</p>

                      <div className="flex flex-wrap gap-2">
                        <Badge variant="outline">
                          <Cpu className="h-3 w-3 mr-1" />
                          {preset.vramEstimate} VRAM
                        </Badge>
                        {preset.config.use_qlora && (
                          <Badge variant="secondary">
                            <Zap className="h-3 w-3 mr-1" />
                            4-bit QLoRA
                          </Badge>
                        )}
                        <Badge variant="outline">
                          Batch: {preset.config.batch_size} × {preset.config.gradient_accumulation_steps}
                        </Badge>
                      </div>
                    </div>

                    {selectedPreset.id === preset.id && (
                      <div className="flex-shrink-0">
                        <div className="h-6 w-6 rounded-full bg-gradient-to-r from-blue-600 to-indigo-600 flex items-center justify-center">
                          <div className="h-2 w-2 rounded-full bg-white" />
                        </div>
                      </div>
                    )}
                  </div>
                </motion.button>
              ))}
            </div>

            {/* VRAM Warning for 8GB */}
            {!selectedPreset.config.use_qlora && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                className="mt-4 p-4 bg-yellow-50 border-2 border-yellow-200 rounded-lg flex items-start gap-3"
              >
                <AlertCircle className="h-5 w-5 text-yellow-600 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-medium text-yellow-900">GPU Memory Warning</p>
                  <p className="text-sm text-yellow-800">
                    This configuration requires {selectedPreset.vramEstimate} VRAM.
                    For 8GB GPUs, we recommend using QLoRA presets.
                  </p>
                </div>
              </motion.div>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Hyperparameters */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <Card className="border-2 border-purple-200">
          <CardHeader>
            <div className="flex items-center gap-3">
              <div className="p-3 bg-gradient-to-r from-purple-500 to-pink-600 rounded-xl shadow-lg">
                <Zap className="h-6 w-6 text-white" />
              </div>
              <div>
                <CardTitle>Hyperparameters</CardTitle>
                <CardDescription>Fine-tune training parameters</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Number of Epochs
                </label>
                <input
                  type="number"
                  value={customConfig.num_epochs}
                  onChange={(e) =>
                    setCustomConfig((prev) => ({
                      ...prev,
                      num_epochs: parseInt(e.target.value),
                    }))
                  }
                  min={1}
                  max={20}
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-purple-500 focus:ring-2 focus:ring-purple-200 transition-all"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Learning Rate
                </label>
                <input
                  type="number"
                  value={customConfig.learning_rate}
                  onChange={(e) =>
                    setCustomConfig((prev) => ({
                      ...prev,
                      learning_rate: parseFloat(e.target.value),
                    }))
                  }
                  step={0.00001}
                  min={0.00001}
                  max={0.001}
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-purple-500 focus:ring-2 focus:ring-purple-200 transition-all"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Warmup Ratio
                </label>
                <input
                  type="number"
                  value={customConfig.warmup_ratio}
                  onChange={(e) =>
                    setCustomConfig((prev) => ({
                      ...prev,
                      warmup_ratio: parseFloat(e.target.value),
                    }))
                  }
                  step={0.01}
                  min={0}
                  max={0.5}
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-purple-500 focus:ring-2 focus:ring-purple-200 transition-all"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Save Every N Steps
                </label>
                <input
                  type="number"
                  value={customConfig.save_steps}
                  onChange={(e) =>
                    setCustomConfig((prev) => ({
                      ...prev,
                      save_steps: parseInt(e.target.value),
                    }))
                  }
                  min={10}
                  step={10}
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-purple-500 focus:ring-2 focus:ring-purple-200 transition-all"
                />
              </div>
            </div>

            {/* Info Box */}
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg flex items-start gap-3">
              <Info className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5" />
              <div className="text-sm text-blue-800">
                <p className="font-medium mb-1">Training Estimate</p>
                <p>
                  With {customConfig.num_epochs} epochs and current settings, training will take
                  approximately{" "}
                  <span className="font-semibold">
                    {Math.round((customConfig.num_epochs * 30) / 60)} - {Math.round((customConfig.num_epochs * 45) / 60)} hours
                  </span>{" "}
                  on a single GPU.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Configuration Summary */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <Card className="bg-gradient-to-r from-green-50 to-emerald-50 border-2 border-green-200">
          <CardHeader>
            <CardTitle className="text-green-900">Configuration Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-600">Model:</span>
                <p className="font-semibold">{selectedPreset.name}</p>
              </div>
              <div>
                <span className="text-gray-600">VRAM:</span>
                <p className="font-semibold">{selectedPreset.vramEstimate}</p>
              </div>
              <div>
                <span className="text-gray-600">Epochs:</span>
                <p className="font-semibold">{customConfig.num_epochs}</p>
              </div>
              <div>
                <span className="text-gray-600">Learning Rate:</span>
                <p className="font-semibold">{customConfig.learning_rate}</p>
              </div>
              <div>
                <span className="text-gray-600">Batch Size:</span>
                <p className="font-semibold">
                  {selectedPreset.config.batch_size} × {selectedPreset.config.gradient_accumulation_steps}{" "}
                  = {selectedPreset.config.batch_size * selectedPreset.config.gradient_accumulation_steps} (effective)
                </p>
              </div>
              <div>
                <span className="text-gray-600">Optimization:</span>
                <p className="font-semibold">
                  {selectedPreset.config.use_qlora ? "4-bit QLoRA" : "LoRA"}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Navigation */}
      <div className="flex gap-3">
        {canGoBack && (
          <Button variant="outline" onClick={onBack} className="flex-1">
            Back to Data Prep
          </Button>
        )}
        <Button onClick={handleComplete} className="flex-1">
          Start Training
        </Button>
      </div>
    </div>
  );
}
