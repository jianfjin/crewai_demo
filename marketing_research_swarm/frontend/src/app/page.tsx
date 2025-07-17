'use client'

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { AnalysisForm } from '@/components/analysis-form'
import { AnalysisMonitor } from '@/components/analysis-monitor'
import { AnalysisResults } from '@/components/analysis-results'
import { apiClient } from '@/lib/api'
import { AnalysisRequest, AnalysisResult } from '@/types/api'
import { Brain, Zap, Users, BarChart3 } from 'lucide-react'

type AppState = 'form' | 'monitoring' | 'results'

export default function Dashboard() {
  const [appState, setAppState] = useState<AppState>('form')
  const [currentAnalysisId, setCurrentAnalysisId] = useState<string | null>(null)
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [isStarting, setIsStarting] = useState(false)

  const handleStartAnalysis = async (request: AnalysisRequest) => {
    setIsStarting(true)
    try {
      const response = await apiClient.startAnalysis(request)
      if (response.analysis_id) {
        setCurrentAnalysisId(response.analysis_id)
      }
      setAppState('monitoring')
    } catch (error) {
      console.error('Failed to start analysis:', error)
      // TODO: Show error toast
    } finally {
      setIsStarting(false)
    }
  }

  const handleAnalysisComplete = (result: AnalysisResult) => {
    setAnalysisResult(result)
    setAppState('results')
  }

  const handleNewAnalysis = () => {
    setAppState('form')
    setCurrentAnalysisId(null)
    setAnalysisResult(null)
  }

  const handleCancel = () => {
    setAppState('form')
    setCurrentAnalysisId(null)
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center w-10 h-10 bg-primary rounded-lg">
                <Brain className="h-6 w-6 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Marketing Research Swarm</h1>
                <p className="text-sm text-gray-500">AI-powered marketing analysis platform</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <div className="hidden md:flex items-center gap-6 text-sm text-gray-600">
                <div className="flex items-center gap-1">
                  <Users className="h-4 w-4" />
                  <span>9 AI Agents</span>
                </div>
                <div className="flex items-center gap-1">
                  <Zap className="h-4 w-4" />
                  <span>Token Optimized</span>
                </div>
                <div className="flex items-center gap-1">
                  <BarChart3 className="h-4 w-4" />
                  <span>Real-time Analytics</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {appState === 'form' && (
          <div className="space-y-8">
            {/* Welcome Section */}
            <Card>
              <CardHeader>
                <CardTitle>Welcome to Marketing Research Swarm</CardTitle>
                <CardDescription>
                  Configure and launch your AI-powered marketing research analysis with our advanced multi-agent system.
                  Choose from predefined analysis types or create custom workflows with your selected agents.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center p-4">
                    <Users className="h-8 w-8 mx-auto mb-2 text-blue-500" />
                    <h3 className="font-semibold mb-1">Multi-Agent System</h3>
                    <p className="text-sm text-gray-600">
                      9 specialized AI agents working together for comprehensive analysis
                    </p>
                  </div>
                  <div className="text-center p-4">
                    <Zap className="h-8 w-8 mx-auto mb-2 text-green-500" />
                    <h3 className="font-semibold mb-1">Token Optimization</h3>
                    <p className="text-sm text-gray-600">
                      Advanced blackboard system reduces token usage by up to 85%
                    </p>
                  </div>
                  <div className="text-center p-4">
                    <BarChart3 className="h-8 w-8 mx-auto mb-2 text-purple-500" />
                    <h3 className="font-semibold mb-1">Real-time Tracking</h3>
                    <p className="text-sm text-gray-600">
                      Monitor progress, token usage, and performance metrics live
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Analysis Configuration */}
            <AnalysisForm 
              onStartAnalysis={handleStartAnalysis}
              isLoading={isStarting}
            />
          </div>
        )}

        {appState === 'monitoring' && currentAnalysisId && (
          <div className="space-y-8">
            <Card>
              <CardHeader>
                <CardTitle>Analysis in Progress</CardTitle>
                <CardDescription>
                  Your marketing research analysis is running. Monitor the progress and token usage in real-time.
                </CardDescription>
              </CardHeader>
            </Card>

            <AnalysisMonitor
              analysisId={currentAnalysisId}
              onComplete={handleAnalysisComplete}
              onCancel={handleCancel}
            />
          </div>
        )}

        {appState === 'results' && analysisResult && (
          <div className="space-y-8">
            <AnalysisResults
              result={analysisResult}
              onNewAnalysis={handleNewAnalysis}
            />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-sm text-gray-500">
            <p>Marketing Research Swarm v1.0.0 - Powered by CrewAI and Next.js</p>
          </div>
        </div>
      </footer>
    </div>
  )
}