'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { apiClient } from '@/lib/api'
import { AnalysisStatus, AnalysisResult, TokenUsage } from '@/types/api'
import { formatDuration, formatTokens, formatCost } from '@/lib/utils'
import { 
  Clock, 
  Zap, 
  CheckCircle, 
  XCircle, 
  AlertCircle, 
  Users, 
  DollarSign,
  Activity,
  StopCircle
} from 'lucide-react'

interface AnalysisMonitorProps {
  analysisId: string
  onComplete: (result: AnalysisResult) => void
  onCancel: () => void
}

export function AnalysisMonitor({ analysisId, onComplete, onCancel }: AnalysisMonitorProps) {
  const [status, setStatus] = useState<AnalysisStatus | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const pollStatus = async () => {
      try {
        const statusData = await apiClient.getAnalysisStatus(analysisId)
        setStatus(statusData)

        if (statusData.status === 'completed' || statusData.status === 'failed') {
          const resultData = await apiClient.getAnalysisResult(analysisId)
          setResult(resultData)
          onComplete(resultData)
          return
        }

        // Continue polling if still running
        if (statusData.status === 'running' || statusData.status === 'starting') {
          setTimeout(pollStatus, 2000) // Poll every 2 seconds
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch status')
      } finally {
        setLoading(false)
      }
    }

    pollStatus()
  }, [analysisId, onComplete])

  const handleCancel = async () => {
    try {
      await apiClient.cancelAnalysis(analysisId)
      onCancel()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cancel analysis')
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'failed':
        return <XCircle className="h-5 w-5 text-red-500" />
      case 'running':
        return <Activity className="h-5 w-5 text-blue-500 animate-pulse" />
      case 'starting':
        return <Clock className="h-5 w-5 text-yellow-500" />
      default:
        return <AlertCircle className="h-5 w-5 text-gray-500" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800'
      case 'failed':
        return 'bg-red-100 text-red-800'
      case 'running':
        return 'bg-blue-100 text-blue-800'
      case 'starting':
        return 'bg-yellow-100 text-yellow-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  if (loading && !status) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
            <span className="ml-2">Loading analysis status...</span>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center gap-2 text-red-600">
            <XCircle className="h-5 w-5" />
            <span>Error: {error}</span>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!status) return null

  return (
    <div className="space-y-6">
      {/* Status Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            {getStatusIcon(status.status)}
            Analysis Status
          </CardTitle>
          <CardDescription>
            Analysis ID: {analysisId}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <Badge className={getStatusColor(status.status)}>
                {status.status.toUpperCase()}
              </Badge>
              {(status.status === 'running' || status.status === 'starting') && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleCancel}
                  className="flex items-center gap-2"
                >
                  <StopCircle className="h-4 w-4" />
                  Cancel
                </Button>
              )}
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Progress</span>
                <span>{Math.round(status.progress)}%</span>
              </div>
              <Progress value={status.progress} className="w-full" />
            </div>

            {status.current_step && (
              <div className="text-sm text-gray-600">
                Current step: {status.current_step}
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Agent Progress */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            Agent Progress
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex justify-between text-sm text-gray-600">
              <span>Completed: {status.agents_completed.length}</span>
              <span>Total: {status.total_agents}</span>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
              {Array.from({ length: status.total_agents }).map((_, index) => {
                const isCompleted = index < status.agents_completed.length
                const agentName = status.agents_completed[index] || `Agent ${index + 1}`
                
                return (
                  <div
                    key={index}
                    className={`p-2 rounded border text-sm ${
                      isCompleted
                        ? 'bg-green-50 border-green-200 text-green-800'
                        : 'bg-gray-50 border-gray-200 text-gray-600'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      {isCompleted ? (
                        <CheckCircle className="h-4 w-4" />
                      ) : (
                        <div className="h-4 w-4 rounded-full border-2 border-gray-300" />
                      )}
                      <span className="truncate">
                        {isCompleted ? agentName.replace('_', ' ') : 'Pending'}
                      </span>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Token Usage (if available) */}
      {status.token_usage && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="h-5 w-5" />
              Token Usage
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {formatTokens(status.token_usage.total_tokens)}
                </div>
                <div className="text-sm text-gray-600">Total Tokens</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {formatTokens(status.token_usage.input_tokens)}
                </div>
                <div className="text-sm text-gray-600">Input Tokens</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">
                  {formatTokens(status.token_usage.output_tokens)}
                </div>
                <div className="text-sm text-gray-600">Output Tokens</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600 flex items-center justify-center gap-1">
                  <DollarSign className="h-5 w-5" />
                  {formatCost(status.token_usage.total_cost)}
                </div>
                <div className="text-sm text-gray-600">Total Cost</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Timing Information */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-5 w-5" />
            Timing
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div>
              <div className="font-medium">Started</div>
              <div className="text-gray-600">
                {new Date(status.start_time).toLocaleString()}
              </div>
            </div>
            <div>
              <div className="font-medium">Duration</div>
              <div className="text-gray-600">
                {formatDuration(Math.floor((new Date().getTime() - new Date(status.start_time).getTime()) / 1000))}
              </div>
            </div>
            {status.estimated_completion && (
              <div>
                <div className="font-medium">Estimated Completion</div>
                <div className="text-gray-600">
                  {new Date(status.estimated_completion).toLocaleString()}
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}