'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Checkbox } from '@/components/ui/checkbox'
import { Slider } from '@/components/ui/slider'
import { MultiSelect } from '@/components/ui/multi-select'
import { Badge } from '@/components/ui/badge'
import { ApiClient } from '@/lib/api'
import { AgentInfo, AnalysisTypeInfo, AnalysisRequest } from '@/types/api'
import { 
  Users, 
  Settings, 
  Target, 
  DollarSign, 
  BarChart3, 
  Zap,
  Brain,
  Clock,
  TrendingUp,
  Loader2
} from 'lucide-react'

interface AnalysisFormProps {
  onStartAnalysis: (request: AnalysisRequest) => void
  isLoading: boolean
}

export function AnalysisForm({ onStartAnalysis, isLoading }: AnalysisFormProps) {
  const [agents, setAgents] = useState<AgentInfo[]>([])
  const [analysisTypes, setAnalysisTypes] = useState<AnalysisTypeInfo[]>([])
  const [loadingData, setLoadingData] = useState(true)
  const [formData, setFormData] = useState<AnalysisRequest>({
    analysis_type: '',
    selected_agents: [],
    optimization_level: 'balanced',
    
    // Campaign Basics
    target_audience: '',
    campaign_type: '',
    budget: 10000,
    duration: '30 days',
    
    // Analysis Focus
    analysis_focus: '',
    business_objective: '',
    competitive_landscape: '',
    
    // Market Segments
    market_segments: [],
    product_categories: [],
    key_metrics: [],
    
    // Brands & Goals
    brands: [],
    campaign_goals: [],
    
    // Forecast & Metrics
    forecast_periods: 12,
    expected_revenue: 100000,
    competitive_analysis: true,
    market_share_analysis: true,
    
    // Brand Metrics
    brand_awareness: 50,
    sentiment_score: 70,
    market_position: 'challenger',
    
    // Optimization Settings
    token_budget: 50000,
    context_strategy: 'adaptive',
    enable_caching: true,
    enable_mem0: true,
    enable_token_tracking: true,
    enable_optimization_tools: true,
    show_comparison: false
  })

  useEffect(() => {
    const loadData = async () => {
      try {
        const [agentsData, typesData] = await Promise.all([
          ApiClient.getAgents(),
          ApiClient.getAnalysisTypes()
        ])
        setAgents(agentsData)
        setAnalysisTypes(typesData)
      } catch (error) {
        console.error('Failed to load form data:', error)
      } finally {
        setLoadingData(false)
      }
    }
    loadData()
  }, [])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onStartAnalysis(formData)
  }

  const handleAnalysisTypeChange = (analysisType: string) => {
    const selectedType = analysisTypes.find(t => t.name === analysisType)
    if (selectedType) {
      setFormData(prev => ({
        ...prev,
        analysis_type: analysisType,
        selected_agents: selectedType.recommended_agents
      }))
    }
  }

  const agentOptions = agents.map(agent => agent.role)

  const marketSegmentOptions = [
    'B2B',
    'B2C', 
    'Enterprise',
    'Small & Medium Business',
    'Consumer',
    'Luxury',
    'Budget'
  ]

  const productCategoryOptions = [
    'Technology',
    'Healthcare',
    'Finance',
    'Retail',
    'Automotive',
    'Food & Beverage',
    'Fashion',
    'Travel'
  ]

  const keyMetricsOptions = [
    'ROI',
    'Conversion Rate',
    'Customer Acquisition Cost',
    'Customer Lifetime Value',
    'Brand Awareness',
    'Engagement Rate',
    'Market Share',
    'Revenue Growth'
  ]

  const campaignGoalOptions = [
    'Brand Awareness',
    'Lead Generation',
    'Sales Conversion',
    'Customer Retention',
    'Market Expansion',
    'Product Launch'
  ]

  if (loadingData) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-12">
          <div className="flex items-center gap-2">
            <Loader2 className="h-6 w-6 animate-spin" />
            <span>Loading analysis configuration...</span>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Analysis Type Selection */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Analysis Configuration
          </CardTitle>
          <CardDescription>
            Choose your analysis type and configure the AI agents for your marketing research.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Analysis Type</label>
              <Select value={formData.analysis_type} onValueChange={handleAnalysisTypeChange}>
                <SelectTrigger>
                  <SelectValue placeholder="Select analysis type" />
                </SelectTrigger>
                <SelectContent>
                  {analysisTypes.map(type => (
                    <SelectItem key={type.name} value={type.name}>
                      <div>
                        <div className="font-medium">{type.name}</div>
                        <div className="text-xs text-gray-500">{type.description}</div>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Optimization Level</label>
              <Select 
                value={formData.optimization_level} 
                onValueChange={(value) => setFormData(prev => ({ ...prev, optimization_level: value }))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="basic">Basic - Standard analysis</SelectItem>
                  <SelectItem value="balanced">Balanced - Optimized performance</SelectItem>
                  <SelectItem value="advanced">Advanced - Maximum optimization</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Selected Agents ({formData.selected_agents.length})</label>
            <MultiSelect
              options={agentOptions}
              value={formData.selected_agents}
              onChange={(value) => setFormData(prev => ({ ...prev, selected_agents: value }))}
              placeholder="Select AI agents for analysis"
            />
            <div className="flex flex-wrap gap-1 mt-2">
              {formData.selected_agents.map(agent => (
                <Badge key={agent} variant="secondary" className="text-xs">
                  {agent}
                </Badge>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Campaign Basics */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-5 w-5" />
            Campaign Basics
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Target Audience</label>
              <Textarea
                value={formData.target_audience}
                onChange={(e) => setFormData(prev => ({ ...prev, target_audience: e.target.value }))}
                placeholder="Describe your target audience..."
                rows={3}
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Campaign Type</label>
              <Select 
                value={formData.campaign_type} 
                onValueChange={(value) => setFormData(prev => ({ ...prev, campaign_type: value }))}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select campaign type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="digital">Digital Marketing</SelectItem>
                  <SelectItem value="social">Social Media</SelectItem>
                  <SelectItem value="content">Content Marketing</SelectItem>
                  <SelectItem value="email">Email Marketing</SelectItem>
                  <SelectItem value="paid">Paid Advertising</SelectItem>
                  <SelectItem value="integrated">Integrated Campaign</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Budget ($)</label>
              <Input
                type="number"
                value={formData.budget}
                onChange={(e) => setFormData(prev => ({ ...prev, budget: parseInt(e.target.value) || 0 }))}
                placeholder="Campaign budget"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Duration</label>
              <Select 
                value={formData.duration} 
                onValueChange={(value) => setFormData(prev => ({ ...prev, duration: value }))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="7 days">1 Week</SelectItem>
                  <SelectItem value="30 days">1 Month</SelectItem>
                  <SelectItem value="90 days">3 Months</SelectItem>
                  <SelectItem value="180 days">6 Months</SelectItem>
                  <SelectItem value="365 days">1 Year</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Analysis Focus */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Analysis Focus
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Analysis Focus</label>
            <Textarea
              value={formData.analysis_focus}
              onChange={(e) => setFormData(prev => ({ ...prev, analysis_focus: e.target.value }))}
              placeholder="What specific aspects should the analysis focus on?"
              rows={2}
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Business Objective</label>
            <Textarea
              value={formData.business_objective}
              onChange={(e) => setFormData(prev => ({ ...prev, business_objective: e.target.value }))}
              placeholder="What are your main business objectives?"
              rows={2}
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Competitive Landscape</label>
            <Textarea
              value={formData.competitive_landscape}
              onChange={(e) => setFormData(prev => ({ ...prev, competitive_landscape: e.target.value }))}
              placeholder="Describe your competitive environment..."
              rows={2}
            />
          </div>
        </CardContent>
      </Card>

      {/* Market Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Market Configuration
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Market Segments</label>
              <MultiSelect
                options={marketSegmentOptions}
                value={formData.market_segments}
                onChange={(value) => setFormData(prev => ({ ...prev, market_segments: value }))}
                placeholder="Select market segments"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Product Categories</label>
              <MultiSelect
                options={productCategoryOptions}
                value={formData.product_categories}
                onChange={(value) => setFormData(prev => ({ ...prev, product_categories: value }))}
                placeholder="Select product categories"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Key Metrics</label>
              <MultiSelect
                options={keyMetricsOptions}
                value={formData.key_metrics}
                onChange={(value) => setFormData(prev => ({ ...prev, key_metrics: value }))}
                placeholder="Select key metrics to track"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Campaign Goals</label>
              <MultiSelect
                options={campaignGoalOptions}
                value={formData.campaign_goals}
                onChange={(value) => setFormData(prev => ({ ...prev, campaign_goals: value }))}
                placeholder="Select campaign goals"
              />
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Brands</label>
            <Input
              value={formData.brands.join(', ')}
              onChange={(e) => setFormData(prev => ({ 
                ...prev, 
                brands: e.target.value.split(',').map(b => b.trim()).filter(b => b) 
              }))}
              placeholder="Enter brand names (comma-separated)"
            />
          </div>
        </CardContent>
      </Card>

      {/* Advanced Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Advanced Settings
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Token Budget</label>
              <div className="space-y-2">
                <Slider
                  value={[formData.token_budget]}
                  onValueChange={(value) => setFormData(prev => ({ ...prev, token_budget: value[0] }))}
                  max={200000}
                  min={10000}
                  step={5000}
                />
                <div className="text-sm text-gray-500">{formData.token_budget.toLocaleString()} tokens</div>
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Forecast Periods</label>
              <Input
                type="number"
                value={formData.forecast_periods}
                onChange={(e) => setFormData(prev => ({ ...prev, forecast_periods: parseInt(e.target.value) || 12 }))}
                min={1}
                max={36}
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Expected Revenue ($)</label>
              <Input
                type="number"
                value={formData.expected_revenue}
                onChange={(e) => setFormData(prev => ({ ...prev, expected_revenue: parseInt(e.target.value) || 0 }))}
                placeholder="Expected revenue"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Market Position</label>
              <Select 
                value={formData.market_position} 
                onValueChange={(value) => setFormData(prev => ({ ...prev, market_position: value }))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="leader">Market Leader</SelectItem>
                  <SelectItem value="challenger">Challenger</SelectItem>
                  <SelectItem value="follower">Follower</SelectItem>
                  <SelectItem value="nicher">Nicher</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Brand Awareness (%)</label>
              <div className="space-y-2">
                <Slider
                  value={[formData.brand_awareness]}
                  onValueChange={(value) => setFormData(prev => ({ ...prev, brand_awareness: value[0] }))}
                  max={100}
                  min={0}
                  step={5}
                />
                <div className="text-sm text-gray-500">{formData.brand_awareness}%</div>
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Sentiment Score (%)</label>
              <div className="space-y-2">
                <Slider
                  value={[formData.sentiment_score]}
                  onValueChange={(value) => setFormData(prev => ({ ...prev, sentiment_score: value[0] }))}
                  max={100}
                  min={0}
                  step={5}
                />
                <div className="text-sm text-gray-500">{formData.sentiment_score}%</div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div className="flex items-center space-x-2">
              <Checkbox
                id="competitive_analysis"
                checked={formData.competitive_analysis}
                onCheckedChange={(checked) => setFormData(prev => ({ ...prev, competitive_analysis: !!checked }))}
              />
              <label htmlFor="competitive_analysis" className="text-sm">Competitive Analysis</label>
            </div>

            <div className="flex items-center space-x-2">
              <Checkbox
                id="market_share_analysis"
                checked={formData.market_share_analysis}
                onCheckedChange={(checked) => setFormData(prev => ({ ...prev, market_share_analysis: !!checked }))}
              />
              <label htmlFor="market_share_analysis" className="text-sm">Market Share Analysis</label>
            </div>

            <div className="flex items-center space-x-2">
              <Checkbox
                id="enable_caching"
                checked={formData.enable_caching}
                onCheckedChange={(checked) => setFormData(prev => ({ ...prev, enable_caching: !!checked }))}
              />
              <label htmlFor="enable_caching" className="text-sm">Enable Caching</label>
            </div>

            <div className="flex items-center space-x-2">
              <Checkbox
                id="enable_mem0"
                checked={formData.enable_mem0}
                onCheckedChange={(checked) => setFormData(prev => ({ ...prev, enable_mem0: !!checked }))}
              />
              <label htmlFor="enable_mem0" className="text-sm">Enable Mem0</label>
            </div>

            <div className="flex items-center space-x-2">
              <Checkbox
                id="enable_token_tracking"
                checked={formData.enable_token_tracking}
                onCheckedChange={(checked) => setFormData(prev => ({ ...prev, enable_token_tracking: !!checked }))}
              />
              <label htmlFor="enable_token_tracking" className="text-sm">Token Tracking</label>
            </div>

            <div className="flex items-center space-x-2">
              <Checkbox
                id="enable_optimization_tools"
                checked={formData.enable_optimization_tools}
                onCheckedChange={(checked) => setFormData(prev => ({ ...prev, enable_optimization_tools: !!checked }))}
              />
              <label htmlFor="enable_optimization_tools" className="text-sm">Optimization Tools</label>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Submit Button */}
      <Card>
        <CardContent className="pt-6">
          <Button 
            type="submit" 
            className="w-full" 
            size="lg"
            disabled={isLoading || !formData.analysis_type || formData.selected_agents.length === 0}
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Starting Analysis...
              </>
            ) : (
              <>
                <Zap className="mr-2 h-4 w-4" />
                Start Marketing Analysis
              </>
            )}
          </Button>
          
          {(!formData.analysis_type || formData.selected_agents.length === 0) && (
            <p className="text-sm text-gray-500 text-center mt-2">
              Please select an analysis type and at least one agent to continue.
            </p>
          )}
        </CardContent>
      </Card>
    </form>
  )
}