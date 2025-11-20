import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import {
  Activity,
  Clock,
  CheckCircle2,
  XCircle,
  Loader2,
  Pause,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { apiClient } from "@/lib/api-client";
import { formatDuration } from "@/lib/utils";

interface TrainingMonitorStepProps {
  trainingConfig: any;
  onComplete: (data: any) => void;
  onBack: () => void;
  canGoBack: boolean;
}

type JobStatus = "pending" | "queued" | "running" | "completed" | "failed" | "cancelled";

export function TrainingMonitorStep({
  trainingConfig,
  onComplete,
  onBack,
  canGoBack,
}: TrainingMonitorStepProps) {
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<JobStatus>("pending");
  const [progress, setProgress] = useState({
    currentEpoch: 0,
    totalEpochs: trainingConfig.num_epochs,
    currentStep: 0,
    totalSteps: 0,
    percentage: 0,
    etaSeconds: 0,
  });
  const [metrics, setMetrics] = useState({
    trainLoss: 0,
    evalLoss: 0,
    learningRate: trainingConfig.learning_rate,
  });
  const [error, setError] = useState<string | null>(null);
  const [startTime, setStartTime] = useState<Date | null>(null);
  const pollingInterval = useRef<NodeJS.Timeout | null>(null);

  // Start training
  useEffect(() => {
    startTraining();
    return () => {
      if (pollingInterval.current) {
        clearInterval(pollingInterval.current);
      }
    };
  }, []);

  const startTraining = async () => {
    try {
      const result = await apiClient.startTraining({
        config: trainingConfig,
        experiment_name: `${trainingConfig.model_name.split("/").pop()}-${new Date().getTime()}`,
        tags: trainingConfig.use_qlora ? ["qlora", "8gb"] : ["lora"],
      }) as { job_id: string };

      setJobId(result.job_id);
      setStartTime(new Date());
      setStatus("running");

      // Start polling for status
      pollingInterval.current = setInterval(pollJobStatus, 2000);
    } catch (err: any) {
      setError(err.message || "Failed to start training");
      setStatus("failed");
    }
  };

  const pollJobStatus = async () => {
    if (!jobId) return;

    try {
      const statusData = await apiClient.getJobStatus(jobId) as {
        status: JobStatus;
        current_epoch?: number;
        total_epochs?: number;
        current_step?: number;
        total_steps?: number;
        progress_percentage?: number;
        eta_seconds?: number;
        train_loss?: number | null;
        eval_loss?: number;
        learning_rate?: number;
        error_message?: string;
      };

      setStatus(statusData.status);
      setProgress({
        currentEpoch: statusData.current_epoch || 0,
        totalEpochs: statusData.total_epochs || trainingConfig.num_epochs,
        currentStep: statusData.current_step || 0,
        totalSteps: statusData.total_steps || 0,
        percentage: statusData.progress_percentage || 0,
        etaSeconds: statusData.eta_seconds || 0,
      });

      if (statusData.train_loss !== null && statusData.train_loss !== undefined) {
        setMetrics((prev) => ({
          ...prev,
          trainLoss: statusData.train_loss!,
          evalLoss: statusData.eval_loss || prev.evalLoss,
          learningRate: statusData.learning_rate || prev.learningRate,
        }));
      }

      // Stop polling if training completed/failed/cancelled
      if (["completed", "failed", "cancelled"].includes(statusData.status)) {
        if (pollingInterval.current) {
          clearInterval(pollingInterval.current);
        }

        if (statusData.status === "completed") {
          setTimeout(() => {
            onComplete({ jobId, status: statusData.status });
          }, 2000);
        }
      }

      if (statusData.error_message) {
        setError(statusData.error_message);
      }
    } catch (err: any) {
      console.error("Polling error:", err);
    }
  };

  const handleCancel = async () => {
    if (!jobId) return;

    try {
      await apiClient.cancelJob(jobId);
      setStatus("cancelled");
      if (pollingInterval.current) {
        clearInterval(pollingInterval.current);
      }
    } catch (err: any) {
      setError(err.message || "Failed to cancel training");
    }
  };

  const elapsedSeconds = startTime
    ? (new Date().getTime() - startTime.getTime()) / 1000
    : 0;

  return (
    <div className="space-y-6">
      {/* Status Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <Card className="border-2 border-blue-200 bg-gradient-to-r from-blue-50 to-indigo-50">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="relative">
                  {status === "running" && (
                    <motion.div
                      className="absolute inset-0 rounded-full bg-blue-400"
                      animate={{
                        scale: [1, 1.3, 1],
                        opacity: [0.5, 0, 0.5],
                      }}
                      transition={{
                        duration: 2,
                        repeat: Infinity,
                        ease: "easeOut",
                      }}
                    />
                  )}
                  <div
                    className={`relative p-4 rounded-full ${
                      status === "running"
                        ? "bg-gradient-to-r from-blue-600 to-indigo-600"
                        : status === "completed"
                        ? "bg-gradient-to-r from-green-600 to-emerald-600"
                        : status === "failed" || status === "cancelled"
                        ? "bg-gradient-to-r from-red-600 to-pink-600"
                        : "bg-gray-400"
                    }`}
                  >
                    {status === "running" ? (
                      <Activity className="h-8 w-8 text-white" />
                    ) : status === "completed" ? (
                      <CheckCircle2 className="h-8 w-8 text-white" />
                    ) : status === "failed" || status === "cancelled" ? (
                      <XCircle className="h-8 w-8 text-white" />
                    ) : (
                      <Loader2 className="h-8 w-8 text-white animate-spin" />
                    )}
                  </div>
                </div>

                <div>
                  <h2 className="text-2xl font-bold text-gray-900">
                    {status === "running"
                      ? "Training in Progress"
                      : status === "completed"
                      ? "Training Completed!"
                      : status === "failed"
                      ? "Training Failed"
                      : status === "cancelled"
                      ? "Training Cancelled"
                      : "Starting Training..."}
                  </h2>
                  <p className="text-gray-600">
                    {jobId ? `Job ID: ${jobId.substring(0, 8)}...` : "Initializing..."}
                  </p>
                </div>
              </div>

              <Badge
                variant={
                  status === "running"
                    ? "default"
                    : status === "completed"
                    ? "success"
                    : status === "failed" || status === "cancelled"
                    ? "destructive"
                    : "outline"
                }
                className="text-lg px-4 py-2"
              >
                {status.toUpperCase()}
              </Badge>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Progress Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <Card>
          <CardHeader>
            <CardTitle>Training Progress</CardTitle>
            <CardDescription>
              Epoch {progress.currentEpoch} of {progress.totalEpochs}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Overall Progress */}
            <div>
              <div className="flex justify-between mb-3">
                <span className="text-sm font-medium text-gray-700">Overall Progress</span>
                <span className="text-sm font-bold text-blue-600">
                  {progress.percentage.toFixed(1)}%
                </span>
              </div>
              <Progress value={progress.percentage} showValue={false} variant="default" />
            </div>

            {/* Steps Progress */}
            {progress.totalSteps > 0 && (
              <div>
                <div className="flex justify-between mb-3">
                  <span className="text-sm font-medium text-gray-700">Current Epoch Steps</span>
                  <span className="text-sm text-gray-600">
                    {progress.currentStep} / {progress.totalSteps}
                  </span>
                </div>
                <Progress
                  value={(progress.currentStep / progress.totalSteps) * 100}
                  showValue={false}
                  variant="success"
                />
              </div>
            )}

            {/* Time Stats */}
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
                <div className="flex items-center gap-2 mb-1">
                  <Clock className="h-4 w-4 text-blue-600" />
                  <span className="text-sm text-gray-600">Elapsed Time</span>
                </div>
                <p className="text-xl font-bold text-blue-900">
                  {formatDuration(elapsedSeconds)}
                </p>
              </div>

              <div className="p-4 bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg border border-purple-200">
                <div className="flex items-center gap-2 mb-1">
                  <Clock className="h-4 w-4 text-purple-600" />
                  <span className="text-sm text-gray-600">ETA</span>
                </div>
                <p className="text-xl font-bold text-purple-900">
                  {progress.etaSeconds > 0
                    ? formatDuration(progress.etaSeconds)
                    : "Calculating..."}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Metrics Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <Card>
          <CardHeader>
            <div className="flex items-center gap-3">
              <TrendingDown className="h-5 w-5 text-green-600" />
              <div>
                <CardTitle>Training Metrics</CardTitle>
                <CardDescription>Loss and learning rate</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-4">
              <div className="p-4 bg-gradient-to-r from-red-50 to-orange-50 rounded-lg border border-red-200">
                <span className="text-sm text-gray-600">Train Loss</span>
                <p className="text-2xl font-bold text-red-900">
                  {metrics.trainLoss > 0 ? metrics.trainLoss.toFixed(4) : "--"}
                </p>
              </div>

              <div className="p-4 bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg border border-green-200">
                <span className="text-sm text-gray-600">Eval Loss</span>
                <p className="text-2xl font-bold text-green-900">
                  {metrics.evalLoss > 0 ? metrics.evalLoss.toFixed(4) : "--"}
                </p>
              </div>

              <div className="p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
                <span className="text-sm text-gray-600">Learning Rate</span>
                <p className="text-2xl font-bold text-blue-900">
                  {metrics.learningRate.toExponential(2)}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Error Display */}
      {error && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="p-4 bg-red-50 border-2 border-red-200 rounded-lg flex items-start gap-3"
        >
          <XCircle className="h-5 w-5 text-red-600 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="font-medium text-red-900">Training Error</p>
            <p className="text-sm text-red-700">{error}</p>
          </div>
        </motion.div>
      )}

      {/* Actions */}
      <div className="flex gap-3">
        {status === "running" && (
          <Button
            variant="destructive"
            onClick={handleCancel}
            className="flex-1"
          >
            <Pause className="h-4 w-4 mr-2" />
            Cancel Training
          </Button>
        )}

        {(status === "failed" || status === "cancelled") && canGoBack && (
          <Button variant="outline" onClick={onBack} className="flex-1">
            Back to Config
          </Button>
        )}

        {status === "completed" && (
          <Button
            onClick={() => onComplete({ jobId, status })}
            className="flex-1"
          >
            Continue to Export
          </Button>
        )}
      </div>
    </div>
  );
}
