# Training UI Components

Modern, animated UI for LLM fine-tuning pipeline built with React, TypeScript, Tailwind CSS, and Framer Motion.

## Design System

- **Framework**: React 18 + TypeScript
- **Styling**: Tailwind CSS 3.3
- **Components**: Shadcn/ui inspired components
- **Animations**: Framer Motion
- **Design Style**: Modern, vibrant (Stripe/Notion-like)
- **Theme**: Light mode with gradient accents

## Components

### 1. TrainingWorkflow (`TrainingWorkflow.tsx`)

Main workflow orchestrator with 4-step wizard:

```tsx
import { TrainingWorkflow } from "@/training";

<TrainingWorkflow />
```

**Features**:
- Animated step indicators with pulse effects
- Progress tracking across steps
- State management for workflow data
- Gradient backgrounds and modern card design

### 2. DataPreparationStep (`DataPreparationStep.tsx`)

Convert SAM2 exports to training format:

**Sub-steps**:
1. Upload & Convert - SAM2 ZIP to LLaVA/HuggingFace format
2. Validate Quality - Check dataset quality and balance
3. Split Dataset - 70/20/10 train/val/test split

**Features**:
- File path inputs with validation
- Format selection (LLaVA/HuggingFace)
- Validation report display with recommendations
- Split statistics

### 3. TrainingConfigStep (`TrainingConfigStep.tsx`)

Configure model and hyperparameters:

**Presets**:
- **LLaVA 7B (QLoRA)** - 8GB GPU optimized ⭐ Recommended
- **LLaVA 7B (LoRA)** - 16GB+ GPU
- **Qwen-VL (QLoRA)** - Alternative model

**Features**:
- Model preset cards with VRAM estimates
- Hyperparameter controls (epochs, learning rate, etc.)
- GPU memory warnings for non-QLoRA configs
- Configuration summary card
- Training time estimates

### 4. TrainingMonitorStep (`TrainingMonitorStep.tsx`)

Real-time training progress monitoring:

**Features**:
- Animated status indicators with pulse effects
- Overall and per-epoch progress bars
- Elapsed time and ETA display
- Live metrics (train loss, eval loss, learning rate)
- Cancel training button
- Auto-polling every 2 seconds
- Error display

### 5. ExportStep (`ExportStep.tsx`)

Export and download trained models:

**Export Formats**:
- **LoRA Adapters** - Small size (~10-50 MB) ⭐ Recommended
- **HuggingFace Full Model** - Merged weights (~13-26 GB)

**Features**:
- Format selection cards
- Export progress
- Download link generation
- Model card generation
- Code examples for loading models

### 6. ExperimentDashboard (`ExperimentDashboard.tsx`)

Track and compare training experiments:

**Features**:
- Stats cards (Total, Completed, Running, Failed)
- Experiment list with sorting
- Status indicators with icons
- Multi-select for comparison (up to 5)
- Delete experiments
- Filter and sort options
- Responsive grid layout

## UI Components Library

### Base Components (`/components/ui/`)

#### Button (`button.tsx`)
Gradient button with variants:
- `default` - Blue gradient with shadow
- `destructive` - Red gradient
- `outline` - Border only
- `secondary` - Purple gradient
- `ghost` - Transparent
- `link` - Underlined text

```tsx
<Button>Click me</Button>
<Button variant="destructive">Delete</Button>
<Button size="lg">Large</Button>
```

#### Card (`card.tsx`)
Modern card with hover effects:

```tsx
<Card>
  <CardHeader>
    <CardTitle>Title</CardTitle>
    <CardDescription>Description</CardDescription>
  </CardHeader>
  <CardContent>Content</CardContent>
  <CardFooter>Footer</CardFooter>
</Card>
```

#### Progress (`progress.tsx`)
Animated progress bar with Framer Motion:

```tsx
<Progress value={75} showValue variant="success" />
```

**Variants**: `default`, `success`, `warning`, `danger`

#### Badge (`badge.tsx`)
Pill-shaped status badges:

```tsx
<Badge variant="success">Completed</Badge>
<Badge variant="warning">Pending</Badge>
```

## Utilities

### API Client (`/lib/api-client.ts`)

Type-safe API client for backend:

```tsx
import { apiClient } from "@/lib/api-client";

// Start training
const result = await apiClient.startTraining({
  config: { ... },
  experiment_name: "my-experiment",
  tags: ["qlora", "8gb"]
});

// Monitor progress
const status = await apiClient.getJobStatus(jobId);

// List experiments
const experiments = await apiClient.listExperiments({
  status: "completed",
  sort_by: "created_at"
});
```

### Utils (`/lib/utils.ts`)

Helper functions:

```tsx
import { cn, formatBytes, formatDuration, calculateETA } from "@/lib/utils";

// Merge Tailwind classes
cn("px-4", "bg-blue-500") // "px-4 bg-blue-500"

// Format file sizes
formatBytes(1024 * 1024 * 500) // "500 MB"

// Format durations
formatDuration(3665) // "1h 1m"

// Calculate ETA
calculateETA(75, 300) // "1m 40s" (based on 75% progress, 300s elapsed)
```

## Color Palette

### Gradients
- **Primary Blue**: `from-blue-600 to-indigo-600`
- **Success Green**: `from-green-500 to-emerald-600`
- **Warning Orange**: `from-yellow-500 to-orange-600`
- **Danger Red**: `from-red-500 to-pink-600`
- **Secondary Purple**: `from-purple-500 to-pink-600`

### Background
- Light gradient: `bg-gradient-to-br from-blue-50 via-white to-purple-50`

## Animations

### Framer Motion Effects

1. **Fade In + Slide Up**:
```tsx
<motion.div
  initial={{ opacity: 0, y: 20 }}
  animate={{ opacity: 1, y: 0 }}
>
```

2. **Staggered Children**:
```tsx
<motion.div
  transition={{ delay: index * 0.1 }}
>
```

3. **Pulse Effect** (for active states):
```tsx
<motion.div
  animate={{ scale: [1, 1.3, 1], opacity: [0.5, 0, 0.5] }}
  transition={{ duration: 2, repeat: Infinity }}
>
```

4. **Scale on Interaction**:
```tsx
<motion.button
  whileHover={{ scale: 1.05 }}
  whileTap={{ scale: 0.95 }}
>
```

## Installation

1. Install dependencies:
```bash
cd demo/frontend
npm install
# Or
yarn install
```

New dependencies added:
- `framer-motion` - Animations
- `lucide-react` - Icons
- `class-variance-authority` - Variant utilities
- `clsx` - Class merging
- `tailwind-merge` - Tailwind class merging

2. Start development server:
```bash
npm run dev
```

3. Start backend API (in separate terminal):
```bash
cd demo/training_api
python main.py
```

## Usage

### Add to Routes

```tsx
// In your routing file (e.g., App.tsx or routes/index.tsx)
import { TrainingWorkflow, ExperimentDashboard } from "@/training";

<Route path="/training" element={<TrainingWorkflow />} />
<Route path="/experiments" element={<ExperimentDashboard />} />
```

### Environment Variables

Create `.env` file in `demo/frontend/`:

```env
VITE_API_URL=http://localhost:8000
```

## Design Principles

1. **Modern & Vibrant** - Bold gradients, generous spacing, rounded corners
2. **Responsive Animations** - Smooth transitions using Framer Motion
3. **User Feedback** - Clear status indicators, loading states, error messages
4. **Accessibility** - Semantic HTML, ARIA labels, keyboard navigation
5. **Performance** - Code splitting, lazy loading, optimized re-renders

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## Screenshots

(Add screenshots of your UI here when deployed)

## Contributing

When adding new components:
1. Follow Shadcn/ui patterns
2. Use Tailwind CSS utilities
3. Add Framer Motion animations
4. Support light mode (dark mode optional)
5. Include TypeScript types
6. Add JSDoc comments

## License

MIT
