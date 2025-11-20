import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  BarChart3,
  Clock,
  CheckCircle2,
  XCircle,
  Loader2,
  Trash2,
  GitCompare,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { apiClient } from "@/lib/api-client";
import { formatDuration } from "@/lib/utils";

interface Experiment {
  experiment_id: string;
  experiment_name: string | null;
  status: string;
  model_name: string;
  best_eval_loss: number | null;
  final_train_loss: number | null;
  num_epochs: number;
  batch_size: number;
  learning_rate: number;
  use_lora: boolean;
  use_qlora: boolean;
  created_at: string;
  duration_seconds: number | null;
  tags: string[];
}

export function ExperimentDashboard() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedForCompare, setSelectedForCompare] = useState<Set<string>>(new Set());
  const [sortBy, setSortBy] = useState<string>("created_at");

  useEffect(() => {
    loadExperiments();
  }, [sortBy]);

  const loadExperiments = async () => {
    setLoading(true);
    try {
      const data = await apiClient.listExperiments({
        limit: 50,
        sort_by: sortBy,
      }) as Experiment[];
      setExperiments(data);
    } catch (error) {
      console.error("Failed to load experiments:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (experimentId: string) => {
    if (!confirm("Are you sure you want to delete this experiment?")) return;

    try {
      await apiClient.deleteExperiment(experimentId);
      await loadExperiments();
    } catch (error: any) {
      alert(`Failed to delete: ${error.message}`);
    }
  };

  const toggleSelectForCompare = (experimentId: string) => {
    const newSelection = new Set(selectedForCompare);
    if (newSelection.has(experimentId)) {
      newSelection.delete(experimentId);
    } else {
      if (newSelection.size < 5) {
        newSelection.add(experimentId);
      }
    }
    setSelectedForCompare(newSelection);
  };

  const handleCompare = () => {
    if (selectedForCompare.size < 2) {
      alert("Please select at least 2 experiments to compare");
      return;
    }
    // TODO: Navigate to comparison view
    console.log("Comparing:", Array.from(selectedForCompare));
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle2 className="h-4 w-4 text-green-600" />;
      case "failed":
      case "cancelled":
        return <XCircle className="h-4 w-4 text-red-600" />;
      case "running":
        return <Loader2 className="h-4 w-4 text-blue-600 animate-spin" />;
      default:
        return <Clock className="h-4 w-4 text-gray-600" />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-6">
      <div className="mx-auto max-w-7xl space-y-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
                Experiment Dashboard
              </h1>
              <p className="text-gray-600 text-lg">
                Track and compare your training experiments
              </p>
            </div>

            <div className="flex gap-3">
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className="px-4 py-2 border-2 border-gray-300 rounded-lg focus:border-blue-500 transition-all bg-white text-gray-900"
                style={{ color: '#111827', backgroundColor: '#ffffff' }}
              >
                <option value="created_at">Recent First</option>
                <option value="duration_seconds">By Duration</option>
                <option value="best_eval_loss">Best Loss</option>
              </select>

              {selectedForCompare.size >= 2 && (
                <Button onClick={handleCompare}>
                  <GitCompare className="h-4 w-4 mr-2" />
                  Compare ({selectedForCompare.size})
                </Button>
              )}
            </div>
          </div>
        </motion.div>

        {/* Stats Cards */}
        <div className="grid grid-cols-4 gap-4">
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-3">
                <div className="p-3 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl">
                  <BarChart3 className="h-6 w-6 text-white" />
                </div>
                <div>
                  <p className="text-sm text-gray-600">Total</p>
                  <p className="text-2xl font-bold">{experiments.length}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-3">
                <div className="p-3 bg-gradient-to-r from-green-500 to-emerald-600 rounded-xl">
                  <CheckCircle2 className="h-6 w-6 text-white" />
                </div>
                <div>
                  <p className="text-sm text-gray-600">Completed</p>
                  <p className="text-2xl font-bold">
                    {experiments.filter((e) => e.status === "completed").length}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-3">
                <div className="p-3 bg-gradient-to-r from-yellow-500 to-orange-600 rounded-xl">
                  <Loader2 className="h-6 w-6 text-white" />
                </div>
                <div>
                  <p className="text-sm text-gray-600">Running</p>
                  <p className="text-2xl font-bold">
                    {experiments.filter((e) => e.status === "running").length}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-3">
                <div className="p-3 bg-gradient-to-r from-red-500 to-pink-600 rounded-xl">
                  <XCircle className="h-6 w-6 text-white" />
                </div>
                <div>
                  <p className="text-sm text-gray-600">Failed</p>
                  <p className="text-2xl font-bold">
                    {experiments.filter((e) => e.status === "failed" || e.status === "cancelled").length}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Experiments List */}
        <div className="space-y-4">
          {loading ? (
            <Card>
              <CardContent className="py-12">
                <div className="flex items-center justify-center gap-3 text-gray-600">
                  <Loader2 className="h-6 w-6 animate-spin" />
                  <span>Loading experiments...</span>
                </div>
              </CardContent>
            </Card>
          ) : experiments.length === 0 ? (
            <Card>
              <CardContent className="py-12">
                <p className="text-center text-gray-500">No experiments yet. Start training to see them here!</p>
              </CardContent>
            </Card>
          ) : (
            experiments.map((exp, index) => (
              <motion.div
                key={exp.experiment_id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <Card
                  className={`transition-all hover:shadow-xl cursor-pointer ${
                    selectedForCompare.has(exp.experiment_id)
                      ? "ring-2 ring-blue-500 bg-blue-50"
                      : ""
                  }`}
                  onClick={() => toggleSelectForCompare(exp.experiment_id)}
                >
                  <CardContent className="p-6">
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1 space-y-3">
                        <div className="flex items-center gap-3">
                          {getStatusIcon(exp.status)}
                          <h3 className="font-bold text-lg">
                            {exp.experiment_name || `Experiment ${exp.experiment_id.substring(0, 8)}`}
                          </h3>
                          <Badge variant={exp.status === "completed" ? "success" : "outline"}>
                            {exp.status}
                          </Badge>
                          {exp.use_qlora && (
                            <Badge variant="secondary">QLoRA</Badge>
                          )}
                        </div>

                        <div className="grid grid-cols-5 gap-4 text-sm">
                          <div>
                            <span className="text-gray-600">Model:</span>
                            <p className="font-medium truncate">{exp.model_name.split("/").pop()}</p>
                          </div>
                          <div>
                            <span className="text-gray-600">Epochs:</span>
                            <p className="font-medium">{exp.num_epochs}</p>
                          </div>
                          <div>
                            <span className="text-gray-600">Train Loss:</span>
                            <p className="font-medium">
                              {exp.final_train_loss?.toFixed(4) || "--"}
                            </p>
                          </div>
                          <div>
                            <span className="text-gray-600">Eval Loss:</span>
                            <p className="font-medium text-green-600">
                              {exp.best_eval_loss?.toFixed(4) || "--"}
                            </p>
                          </div>
                          <div>
                            <span className="text-gray-600">Duration:</span>
                            <p className="font-medium">
                              {exp.duration_seconds ? formatDuration(exp.duration_seconds) : "--"}
                            </p>
                          </div>
                        </div>

                        <div className="flex gap-2">
                          {exp.tags.map((tag) => (
                            <Badge key={tag} variant="outline" className="text-xs">
                              {tag}
                            </Badge>
                          ))}
                        </div>
                      </div>

                      <div className="flex flex-col gap-2">
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDelete(exp.experiment_id);
                          }}
                        >
                          <Trash2 className="h-4 w-4 text-red-600" />
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
