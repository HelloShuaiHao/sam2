import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Database,
  Settings,
  Play,
  Download,
  CheckCircle2,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

// Import step components (will be created next)
import { DataPreparationStep } from "./DataPreparationStep";
import { TrainingConfigStep } from "./TrainingConfigStep";
import { TrainingMonitorStep } from "./TrainingMonitorStep";
import { ExportStep } from "./ExportStep";

const steps = [
  {
    id: "data-prep",
    title: "Data Preparation",
    description: "Convert and validate your dataset",
    icon: Database,
    component: DataPreparationStep,
  },
  {
    id: "config",
    title: "Training Config",
    description: "Configure model and hyperparameters",
    icon: Settings,
    component: TrainingConfigStep,
  },
  {
    id: "train",
    title: "Train Model",
    description: "Monitor training progress",
    icon: Play,
    component: TrainingMonitorStep,
  },
  {
    id: "export",
    title: "Export Model",
    description: "Download your trained model",
    icon: Download,
    component: ExportStep,
  },
];

interface WorkflowState {
  currentStep: number;
  completedSteps: Set<number>;
  dataConfig: any;
  trainingConfig: any;
  jobId?: string;
}

export function TrainingWorkflow() {
  const [state, setState] = useState<WorkflowState>({
    currentStep: 0,
    completedSteps: new Set(),
    dataConfig: null,
    trainingConfig: null,
  });

  const currentStep = steps[state.currentStep];
  const CurrentStepComponent = currentStep.component;

  const goToStep = (stepIndex: number) => {
    if (stepIndex <= state.completedSteps.size) {
      setState((prev) => ({ ...prev, currentStep: stepIndex }));
    }
  };

  const completeStep = (data?: any) => {
    setState((prev) => {
      const newCompleted = new Set(prev.completedSteps);
      newCompleted.add(prev.currentStep);

      // Save step data
      const updates: Partial<WorkflowState> = {
        completedSteps: newCompleted,
      };

      if (prev.currentStep === 0) {
        updates.dataConfig = data;
      } else if (prev.currentStep === 1) {
        updates.trainingConfig = data;
      } else if (prev.currentStep === 2) {
        updates.jobId = data?.jobId;
      }

      // Move to next step if not on last step
      if (prev.currentStep < steps.length - 1) {
        updates.currentStep = prev.currentStep + 1;
      }

      return { ...prev, ...updates };
    });
  };

  const goBack = () => {
    if (state.currentStep > 0) {
      setState((prev) => ({ ...prev, currentStep: prev.currentStep - 1 }));
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-6">
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
            LLM Fine-tuning Workflow
          </h1>
          <p className="text-gray-600 text-lg">
            Train your vision-language model in 4 easy steps
          </p>
        </motion.div>

        {/* Progress Steps */}
        <Card className="mb-8 overflow-hidden">
          <CardContent className="p-0">
            <div className="flex items-center justify-between p-6">
              {steps.map((step, index) => {
                const Icon = step.icon;
                const isCompleted = state.completedSteps.has(index);
                const isCurrent = state.currentStep === index;
                const isAccessible = index <= state.completedSteps.size;

                return (
                  <React.Fragment key={step.id}>
                    {/* Step Button */}
                    <motion.button
                      onClick={() => goToStep(index)}
                      disabled={!isAccessible}
                      className={cn(
                        "flex flex-col items-center gap-2 transition-all group relative",
                        isAccessible ? "cursor-pointer" : "cursor-not-allowed opacity-50"
                      )}
                      whileHover={isAccessible ? { scale: 1.05 } : {}}
                      whileTap={isAccessible ? { scale: 0.95 } : {}}
                    >
                      {/* Icon Circle */}
                      <div
                        className={cn(
                          "relative flex h-14 w-14 items-center justify-center rounded-full transition-all",
                          isCompleted
                            ? "bg-gradient-to-r from-green-500 to-emerald-600 shadow-lg shadow-green-500/50"
                            : isCurrent
                            ? "bg-gradient-to-r from-blue-600 to-indigo-600 shadow-xl shadow-blue-500/50 ring-4 ring-blue-100"
                            : "bg-gray-200 group-hover:bg-gray-300"
                        )}
                      >
                        {isCompleted ? (
                          <CheckCircle2 className="h-7 w-7 text-white" />
                        ) : (
                          <Icon
                            className={cn(
                              "h-7 w-7",
                              isCurrent ? "text-white" : "text-gray-500"
                            )}
                          />
                        )}

                        {/* Pulse animation for current step */}
                        {isCurrent && (
                          <motion.div
                            className="absolute inset-0 rounded-full bg-blue-400"
                            initial={{ scale: 1, opacity: 0.5 }}
                            animate={{ scale: 1.3, opacity: 0 }}
                            transition={{
                              duration: 1.5,
                              repeat: Infinity,
                              ease: "easeOut",
                            }}
                          />
                        )}
                      </div>

                      {/* Step Info */}
                      <div className="text-center">
                        <p
                          className={cn(
                            "text-sm font-semibold",
                            isCurrent
                              ? "text-blue-600"
                              : isCompleted
                              ? "text-green-600"
                              : "text-gray-500"
                          )}
                        >
                          {step.title}
                        </p>
                        <p className="text-xs text-gray-500 max-w-[120px]">
                          {step.description}
                        </p>
                      </div>

                      {/* Step Number Badge */}
                      <Badge
                        variant={
                          isCompleted ? "success" : isCurrent ? "default" : "outline"
                        }
                        className="absolute -top-2 -right-2"
                      >
                        {index + 1}
                      </Badge>
                    </motion.button>

                    {/* Connector Line */}
                    {index < steps.length - 1 && (
                      <div className="flex-1 px-4">
                        <div className="relative h-1 bg-gray-200 rounded-full overflow-hidden">
                          <motion.div
                            className="absolute inset-y-0 left-0 bg-gradient-to-r from-green-500 to-emerald-600"
                            initial={{ width: "0%" }}
                            animate={{
                              width: state.completedSteps.has(index) ? "100%" : "0%",
                            }}
                            transition={{ duration: 0.5 }}
                          />
                        </div>
                      </div>
                    )}
                  </React.Fragment>
                );
              })}
            </div>
          </CardContent>
        </Card>

        {/* Step Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={currentStep.id}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
          >
            <CurrentStepComponent
              dataConfig={state.dataConfig}
              trainingConfig={state.trainingConfig}
              jobId={state.jobId}
              onComplete={completeStep}
              onBack={goBack}
              canGoBack={state.currentStep > 0}
            />
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
