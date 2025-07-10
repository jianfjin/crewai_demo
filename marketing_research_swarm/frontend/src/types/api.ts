export interface AgentInfo {
  role: string
  goal: string
  backstory: string
  tools: string[]
  phase: string
}

export interface AnalysisTypeInfo {
  name: string
  description: string
  recommended_agents: string[]
  estimated_duration: number
  complexity: string
}

export interface AnalysisRequest {
  analysis_type: string
  selected_agents: string[]
  optimization_level: string
  custom_inputs?: Record<string, any>
}

export interface AnalysisResponse {
  analysis_id: string
  status: string
  message: string
  estimated_duration?: number
}

export interface AnalysisStatus {
  analysis_id: string
  status: string
  progress: number
  current_step?: string
  agents_completed: string[]
  total_agents: number
  start_time: string
  estimated_completion?: string
  token_usage?: TokenUsage
}

export interface AnalysisResult {
  analysis_id: string
  status: string
  result?: string
  token_usage?: TokenUsage
  performance_metrics?: Record<string, any>
  error_message?: string
  duration?: number
}

export interface TokenUsage {
  total_tokens: number
  input_tokens: number
  output_tokens: number
  total_cost: number
  model_used?: string
  agent_breakdown?: Record<string, AgentTokenUsage>
}

export interface AgentTokenUsage {
  total_tokens: number
  input_tokens: number
  output_tokens: number
  cost: number
  tasks: Record<string, TaskTokenUsage>
}

export interface TaskTokenUsage {
  tokens: number
  duration: number
}

export interface AnalysisHistoryItem {
  analysis_id: string
  status: string
  start_time: string
  end_time?: string
  duration?: number
  analysis_type: string
  agents_count: number
  token_usage: number
}