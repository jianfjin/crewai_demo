'use client'

import React from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { AnalysisResult, TokenUsage } from '@/types/api'
import { formatDuration, formatTokens, formatCost } from '@/lib/utils'
import { 
  CheckCircle, 
  XCircle, 
  Download, 
  Zap, 
  Clock, 
  DollarSign, 
  BarChart3,
  FileText,
  Users
} from 'lucide-react'

interface AnalysisResultsProps {
  result: AnalysisResult
  onNewAnalysis: () => void
}

export function AnalysisResults({ result, onNewAnalysis }: AnalysisResultsProps) {
  const handleDownload = () => {
    if (result.result) {
      const blob = new Blob([result.result], { type: 'text/markdown' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `analysis-${result.analysis_id}.md`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    }
  }

  const getStatusIcon = () => {
    switch (result.status) {
      case 'completed':
        return <CheckCircle className="h-6 w-6 text-green-500" />
      case 'failed':
        return <XCircle className="h-6 w-6 text-red-500" />
      default:
        return null
    }
  }

  const getStatusColor = () => {
    switch (result.status) {
      case 'completed':
        return 'bg-green-100 text-green-800'
      case 'failed':
        return 'bg-red-100 text-red-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  return (
    <div className="space-y-6">
      {/* Status Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {getStatusIcon()}
              Analysis {result.status === 'completed' ? 'Completed' : 'Failed'}
            </div>
            <div className="flex items-center gap-2">
              <Badge className={getStatusColor()}>
                {result.status.toUpperCase()}
              </Badge>
              <Button onClick={onNewAnalysis} variant="outline">
                New Analysis
              </Button>
            </div>
          </CardTitle>
          <CardDescription>
            Analysis ID: {result.analysis_id}
          </CardDescription>
        </CardHeader>
      </Card>

      {/* Error Message (if failed) */}
      {result.status === 'failed' && result.error_message && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-red-600">
              <XCircle className="h-5 w-5" />
              Error Details
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <p className="text-red-800">{result.error_message}</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Performance Metrics */}
      {result.duration && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Performance Metrics
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600 flex items-center justify-center gap-1">
                  <Clock className="h-5 w-5" />
                  {formatDuration(result.duration)}
                </div>
                <div className="text-sm text-gray-600">Total Duration</div>
              </div>
              
              {result.token_usage && (
                <>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600 flex items-center justify-center gap-1">
                      <Zap className="h-5 w-5" />
                      {formatTokens(result.token_usage.total_tokens)}
                    </div>
                    <div className="text-sm text-gray-600">Total Tokens</div>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-2xl font-bold text-orange-600 flex items-center justify-center gap-1">
                      <DollarSign className="h-5 w-5" />
                      {formatCost(result.token_usage.total_cost)}
                    </div>
                    <div className="text-sm text-gray-600">Total Cost</div>
                  </div>
                </>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Token Usage Breakdown */}
      {result.token_usage && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="h-5 w-5" />
              Token Usage Breakdown
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Overall Usage */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-gray-50 rounded-lg">
                <div className="text-center">
                  <div className="text-lg font-semibold text-blue-600">
                    {formatTokens(result.token_usage.total_tokens)}
                  </div>
                  <div className="text-xs text-gray-600">Total</div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-semibold text-green-600">
                    {formatTokens(result.token_usage.input_tokens)}
                  </div>
                  <div className="text-xs text-gray-600">Input</div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-semibold text-purple-600">
                    {formatTokens(result.token_usage.output_tokens)}
                  </div>
                  <div className="text-xs text-gray-600">Output</div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-semibold text-orange-600">
                    {formatCost(result.token_usage.total_cost)}
                  </div>
                  <div className="text-xs text-gray-600">Cost</div>
                </div>
              </div>

              {/* Agent Breakdown */}
              {result.token_usage.agent_breakdown && (
                <div className="space-y-3">
                  <h4 className="font-medium flex items-center gap-2">
                    <Users className="h-4 w-4" />
                    Agent Breakdown
                  </h4>
                  <div className="space-y-2">
                    {Object.entries(result.token_usage.agent_breakdown).map(([agent, usage]) => (
                      <div key={agent} className="border rounded-lg p-3">
                        <div className="flex items-center justify-between mb-2">
                          <h5 className="font-medium text-sm">
                            {agent.replace('_', ' ').toUpperCase()}
                          </h5>
                          <Badge variant="outline">
                            {formatCost(usage.cost)}
                          </Badge>
                        </div>
                        <div className="grid grid-cols-3 gap-2 text-xs">
                          <div className="text-center">
                            <div className="font-medium">{formatTokens(usage.total_tokens)}</div>
                            <div className="text-gray-600">Total</div>
                          </div>
                          <div className="text-center">
                            <div className="font-medium">{formatTokens(usage.input_tokens)}</div>
                            <div className="text-gray-600">Input</div>
                          </div>
                          <div className="text-center">
                            <div className="font-medium">{formatTokens(usage.output_tokens)}</div>
                            <div className="text-gray-600">Output</div>
                          </div>
                        </div>
                        
                        {/* Task Breakdown */}
                        {usage.tasks && Object.keys(usage.tasks).length > 0 && (
                          <div className="mt-2 pt-2 border-t">
                            <div className="text-xs text-gray-600 mb-1">Tasks:</div>
                            <div className="space-y-1">
                              {Object.entries(usage.tasks).map(([task, taskUsage]) => (
                                <div key={task} className="flex justify-between text-xs">
                                  <span className="truncate">{task.replace('_', ' ')}</span>
                                  <span>{formatTokens(taskUsage.tokens)} ({formatDuration(taskUsage.duration)})</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Analysis Results */}
      {result.result && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                Analysis Results
              </div>
              <Button onClick={handleDownload} variant="outline" size="sm">
                <Download className="h-4 w-4 mr-2" />
                Download
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="bg-gray-50 rounded-lg p-4 max-h-96 overflow-y-auto">
              <pre className="whitespace-pre-wrap text-sm font-mono">
                {result.result}
              </pre>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Performance Metrics Details */}
      {result.performance_metrics && Object.keys(result.performance_metrics).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Detailed Performance Metrics
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="bg-gray-50 rounded-lg p-4">
              <pre className="text-sm">
                {JSON.stringify(result.performance_metrics, null, 2)}
              </pre>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}