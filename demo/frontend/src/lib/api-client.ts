/**
 * API client for training backend
 * Uses relative URL to go through nginx proxy at /api/training/
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || "/api/training";

export interface ApiError {
  error: string;
  message: string;
  type?: string;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options?: RequestInit
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          "Content-Type": "application/json",
          ...options?.headers,
        },
      });

      if (!response.ok) {
        const error: ApiError = await response.json();
        throw new Error(error.message || "API request failed");
      }

      return await response.json();
    } catch (error) {
      console.error("API request failed:", error);
      throw error;
    }
  }

  // Data preparation endpoints
  async convertData(data: {
    sam2_zip_path: string;
    output_dir: string;
    target_format: "huggingface" | "llava";
  }) {
    return this.request("/data/convert", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async validateData(data: {
    data_path: string;
    format_type: "huggingface" | "llava";
  }) {
    return this.request("/data/validate", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async splitData(data: {
    data_path: string;
    output_dir: string;
    strategy: "stratified" | "temporal" | "random";
    train_ratio: number;
    val_ratio: number;
    test_ratio: number;
    random_seed: number;
  }) {
    return this.request("/data/split", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  // Training endpoints
  async startTraining(data: {
    config: any;
    experiment_name?: string;
    tags?: string[];
  }) {
    return this.request("/train/start", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async getJobStatus(jobId: string) {
    return this.request(`/train/${jobId}/status`);
  }

  async cancelJob(jobId: string) {
    return this.request(`/train/${jobId}/cancel`, {
      method: "POST",
    });
  }

  async listJobs(status?: string, limit: number = 100) {
    const params = new URLSearchParams();
    if (status) params.append("status", status);
    params.append("limit", limit.toString());

    return this.request(`/train/jobs?${params.toString()}`);
  }

  // Experiment endpoints
  async listExperiments(filters?: {
    status?: string;
    tag?: string;
    limit?: number;
    sort_by?: string;
  }) {
    const params = new URLSearchParams();
    if (filters?.status) params.append("status", filters.status);
    if (filters?.tag) params.append("tag", filters.tag);
    if (filters?.limit) params.append("limit", filters.limit.toString());
    if (filters?.sort_by) params.append("sort_by", filters.sort_by);

    return this.request(`/experiments?${params.toString()}`);
  }

  async getExperiment(experimentId: string) {
    return this.request(`/experiments/${experimentId}`);
  }

  async compareExperiments(data: {
    experiment_ids: string[];
    metrics: string[];
  }) {
    return this.request("/experiments/compare", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async deleteExperiment(experimentId: string) {
    return this.request(`/experiments/${experimentId}`, {
      method: "DELETE",
    });
  }

  async getExperimentMetrics(experimentId: string) {
    return this.request(`/experiments/${experimentId}/metrics`);
  }

  // Export endpoints
  async exportModel(jobId: string, data: {
    export_format: "huggingface" | "lora_adapter" | "onnx" | "tflite";
    output_dir?: string;
    generate_model_card?: boolean;
    merge_adapters?: boolean;
  }) {
    return this.request(`/export/${jobId}`, {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async getDownloadInfo(jobId: string) {
    return this.request(`/export/${jobId}/download`);
  }

  async listExports() {
    return this.request("/export/list");
  }

  async deleteExport(exportId: string) {
    return this.request(`/export/${exportId}`, {
      method: "DELETE",
    });
  }
}

export const apiClient = new ApiClient();
