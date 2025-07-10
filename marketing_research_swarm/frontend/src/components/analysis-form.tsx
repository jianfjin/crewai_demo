'use client'

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { ApiClient } from '@/lib/api'
import { AgentInfo, AnalysisTypeInfo, AnalysisRequest } from '@/types/api'
import { Play, Users, Clock, Zap } from 'lucide-react'

interface AnalysisFormProps {
  onStartAnalysis: (request: AnalysisRequest) => void
  isLoading: boolean
}

export function AnalysisForm({ onStartAnalysis, isLoading }: AnalysisFormProps) {
  const [agents, setAgents] = useState<AgentInfo[]>([])
  const [analysisTypes, setAnalysisTypes] = useState<AnalysisTypeInfo[]>([])
  const [selectedType, setSelectedType] = useState<string>('')
  const [selectedAgents, setSelectedAgents] = useState<string[]>([])
  const [optimizationLevel, setOptimizationLevel] = useState<string>('blackboard')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    try {
      const [agentsData, typesData] = await Promise.all([
        ApiClient.getAgents(),
        ApiClient.getAnalysisTypes()
      ])
      setAgents(agentsData)
      setAnalysisTypes(typesData)
      
      // Set default analysis type
      if (typesData.length > 0) {
        setSelectedType(typesData[0].name)
        setSelectedAgents(typesData[0].recommended_agents.slice(0, 3))
      }
    } catch (error) {
      console.error('Failed to load data:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleTypeChange = (typeName: string) => {
    setSelectedType(typeName)
    const type = analysisTypes.find(t => t.name === typeName)
    if (type) {
      setSelectedAgents(type.recommended_agents.slice(0, 3))
    }
  }

  const toggleAgent = (agentRole: string) => {
    setSelectedAgents(prev => 
      prev.includes(agentRole)
        ? prev.filter(a => a !== agentRole)
        : [...prev, agentRole]
    )
  }

  const handleSubmit = () => {
    if (selectedType && selectedAgents.length > 0) {
      const request: AnalysisRequest = {
        analysis_type: selectedType,
        selected_agents: selectedAgents,
        optimization_level: optimizationLevel,
        custom_inputs: {}
      }
      onStartAnalysis(request)
    }
  }

  const getPhaseColor = (phase: string) => {
    switch (phase) {
      case 'foundation': return 'bg-blue-100 text-blue-800'
      case 'analysis': return 'bg-green-100 text-green-800'
      case 'strategy': return 'bg-purple-100 text-purple-800'
      case 'content': return 'bg-orange-100 text-orange-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  if (loading) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
            <span className="ml-2">Loading analysis options...</span>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      {/* Analysis Type Selection */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Analysis Type
          </CardTitle>
          <CardDescription>
            Choose the type of marketing research analysis you want to perform
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {analysisTypes.map((type) => (
              <div
                key={type.name}
                className={`p-4 border rounded-lg cursor-pointer transition-all ${
                  selectedType === type.name
                    ? 'border-primary bg-primary/5'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => handleTypeChange(type.name)}
              >
                <h3 className="font-semibold mb-2">{type.name.replace('_', ' ').toUpperCase()}</h3>
                <p className="text-sm text-gray-600 mb-3">{type.description}</p>
                <div className="flex items-center justify-between text-xs text-gray-500">
                  <span className="flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    {Math.floor(type.estimated_duration / 60)}m
                  </span>
                  <Badge variant={type.complexity === 'High' ? 'destructive' : type.complexity === 'Medium' ? 'default' : 'secondary'}>
                    {type.complexity}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Agent Selection */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            Agent Selection
          </CardTitle>
          <CardDescription>
            Select the AI agents that will work on your analysis ({selectedAgents.length} selected)
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {agents.map((agent) => (
              <div
                key={agent.role}
                className={`p-4 border rounded-lg cursor-pointer transition-all ${
                  selectedAgents.includes(agent.role)
                    ? 'border-primary bg-primary/5'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => toggleAgent(agent.role)}
              >
                <div className="flex items-start justify-between mb-2">
                  <h3 className="font-semibold text-sm">{agent.role.replace('_', ' ').toUpperCase()}</h3>
                  <Badge className={getPhaseColor(agent.phase)}>
                    {agent.phase}
                  </Badge>
                </div>
                <p className="text-xs text-gray-600 mb-2">{agent.goal}</p>
                <div className="flex flex-wrap gap-1">
                  {agent.tools.slice(0, 3).map((tool) => (
                    <Badge key={tool} variant="outline" className="text-xs">
                      {tool.replace('_', ' ')}
                    </Badge>
                  ))}
                  {agent.tools.length > 3 && (
                    <Badge variant="outline" className="text-xs">
                      +{agent.tools.length - 3}
                    </Badge>
                  )}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Optimization Level */}
      <Card>
        <CardHeader>
          <CardTitle>Optimization Level</CardTitle>
          <CardDescription>
            Choose the optimization strategy for token efficiency and performance
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {[
              {
                value: 'none',
                title: 'No Optimization',
                description: 'Standard execution without optimization',
                efficiency: '0%'
              },
              {
                value: 'partial',
                title: 'Partial Optimization',
                description: 'Basic context management and caching',
                efficiency: '30%'
              },
              {
                value: 'blackboard',
                title: 'Blackboard System',
                description: 'Advanced shared state and context isolation',
                efficiency: '85%'
              }
            ].map((option) => (
              <div
                key={option.value}
                className={`p-4 border rounded-lg cursor-pointer transition-all ${
                  optimizationLevel === option.value
                    ? 'border-primary bg-primary/5'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => setOptimizationLevel(option.value)}
              >
                <h3 className="font-semibold mb-1">{option.title}</h3>
                <p className="text-sm text-gray-600 mb-2">{option.description}</p>
                <Badge variant={option.value === 'blackboard' ? 'default' : 'secondary'}>
                  {option.efficiency} Token Reduction
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Start Analysis Button */}
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-semibold">Ready to Start Analysis</h3>
              <p className="text-sm text-gray-600">
                {selectedAgents.length} agents selected • {optimizationLevel} optimization
              </p>
            </div>
            <Button
              onClick={handleSubmit}
              disabled={isLoading || selectedAgents.length === 0}
              className="flex items-center gap-2"
            >
              <Play className="h-4 w-4" />
              {isLoading ? 'Starting...' : 'Start Analysis'}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}