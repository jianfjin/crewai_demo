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
    optimization_level: 'blackboard',
    
    // Campaign Basics - Match streamlit defaults
    target_audience: 'health-conscious millennials and premium beverage consumers',
    campaign_type: 'multi-channel global marketing campaign',
    budget: 250000,
    duration: '12 months',
    
    // Analysis Focus - Match streamlit defaults
    analysis_focus: 'global beverage market performance and brand optimization',
    business_objective: 'Optimize beverage portfolio performance across global markets and develop data-driven marketing strategies',
    competitive_landscape: 'global beverage market with diverse categories including Cola, Juice, Energy, Sports drinks, and enhanced water',
    
    // Market Segments - Match streamlit defaults
    market_segments: ['North America', 'Europe', 'Asia Pacific'],
    product_categories: ['Cola', 'Juice', 'Energy', 'Sports'],
    key_metrics: ['brand_performance', 'category_trends', 'profitability_analysis'],
    
    // Brands & Goals - Match streamlit defaults
    brands: ['Coca-Cola', 'Pepsi', 'Red Bull'],
    campaign_goals: [
      'Optimize brand portfolio performance across global markets',
      'Identify high-margin opportunities by category and region',
      'Develop pricing strategies based on profitability analysis'
    ],
    
    // Forecast & Metrics - Match streamlit defaults
    forecast_periods: 30,
    expected_revenue: 25000,
    competitive_analysis: true,
    market_share_analysis: true,
    
    // Brand Metrics - Match streamlit defaults
    brand_awareness: 75,
    sentiment_score: 60,
    market_position: 'Leader',
    
    // Optimization Settings - Match streamlit defaults
    token_budget: 4000,
    context_strategy: 'progressive_pruning',
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

  // Match streamlit dashboard options exactly
  const marketSegmentOptions = [
    'North America',
    'Europe', 
    'Asia Pacific',
    'Latin America',
    'Middle East',
    'Africa',
    'Australia',
    'Global'
  ]

  const productCategoryOptions = [
    'Cola',
    'Juice',
    'Energy',
    'Sports',
    'Citrus',
    'Lemon-Lime',
    'Orange',
    'Water',
    'Enhanced Water',
    'Tea',
    'Coffee'
  ]

  const keyMetricsOptions = [
    'brand_performance',
    'category_trends',
    'regional_dynamics',
    'profitability_analysis',
    'pricing_optimization',
    'market_share',
    'customer_satisfaction',
    'roi'
  ]

  const campaignGoalOptions = [
    'Optimize brand portfolio performance across global markets',
    'Identify high-margin opportunities by category and region',
    'Develop pricing strategies based on profitability analysis',
    'Create targeted marketing strategies for different beverage categories',
    'Forecast sales and revenue for strategic planning',
    'Enhance brand positioning in competitive categories',
    'Increase market share in key segments',
    'Improve customer acquisition and retention'
  ]

  const campaignTypeOptions = [
    'multi-channel global marketing campaign',
    'digital marketing campaign',
    'traditional media campaign',
    'social media campaign',
    'influencer marketing campaign'
  ]

  const durationOptions = [
    '3 months',
    '6 months',
    '12 months',
    '18 months',
    '24 months'
  ]

  const marketPositionOptions = [
    'Leader',
    'Challenger',
    'Follower',
    'Niche'
  ]

  const optimizationLevelOptions = [
    { value: 'blackboard', label: 'Blackboard System', description: '85-95% token reduction expected' },
    { value: 'full', label: 'Full Optimization', description: '75-85% token reduction expected' },
    { value: 'partial', label: 'Partial Optimization', description: '40-50% token reduction expected' },
    { value: 'none', label: 'No Optimization', description: 'Standard token usage (baseline)' }
  ]

  const contextStrategyOptions = [
    'progressive_pruning',
    'abstracted_summaries',
    'minimal_context',
    'stateless'
  ]

  const brandOptions = [
    'Coca-Cola',
    'Pepsi',
    'Red Bull',
    'Monster Energy',
    'Gatorade',
    'Powerade',
    'Tropicana',
    'Simply Orange',
    'Minute Maid',
    'Sprite',
    'Fanta',
    '7UP',
    'Mountain Dew',
    'Dr Pepper',
    'Dasani Water',
    'Aquafina',
    'Vitamin Water'
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
      {/* Analysis Configuration */}
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
                        <div className="font-medium">{type.name.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</div>
                        <div className="text-xs text-gray-500">{type.description}</div>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {formData.analysis_type && (
                <div className="text-xs text-gray-600 mt-1">
                  {analysisTypes.find(t => t.name === formData.analysis_type)?.description}
                </div>
              )}
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
                  {optimizationLevelOptions.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      <div>
                        <div className="font-medium">{option.label}</div>
                        <div className="text-xs text-gray-500">{option.description}</div>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {formData.optimization_level && (
                <div className="text-xs text-gray-600 mt-1">
                  {optimizationLevelOptions.find(o => o.value === formData.optimization_level)?.description}
                </div>
              )}
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Agent Phase Selection</label>
            <div className="p-3 border rounded-lg bg-gray-50">
              <div className="text-sm text-gray-600 mb-2">
                <strong>Agent Phases:</strong>
              </div>
              <div className="text-xs text-gray-500 mb-3">
                <strong>FOUNDATION:</strong> market_research_analyst, data_analyst<br/>
                <strong>ANALYSIS:</strong> competitive_analyst, brand_performance_specialist<br/>
                <strong>STRATEGY:</strong> brand_strategist, campaign_optimizer, forecasting_specialist<br/>
                <strong>CONTENT:</strong> content_strategist, creative_copywriter
              </div>
              <MultiSelect
                options={agentOptions}
                value={formData.selected_agents}
                onChange={(value) => setFormData(prev => ({ ...prev, selected_agents: value }))}
                placeholder="Choose agents by phase or select optimal analysis types"
              />
              <div className="flex flex-wrap gap-1 mt-2">
                {formData.selected_agents.map(agent => (
                  <Badge key={agent} variant="secondary" className="text-xs">
                    {agent}
                  </Badge>
                ))}
              </div>
              {formData.analysis_type !== 'custom' && formData.selected_agents.length > 0 && (
                <div className="text-xs text-green-600 mt-2">
                  ✅ Auto-selected {formData.selected_agents.length} agents for {formData.analysis_type.replace('_', ' ')} analysis
                </div>
              )}
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
                placeholder="Describe your target audience"
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
                  {campaignTypeOptions.map((option) => (
                    <SelectItem key={option} value={option}>
                      {option}
                    </SelectItem>
                  ))}
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
                min={1000}
                max={10000000}
                step={1000}
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
                  {durationOptions.map((option) => (
                    <SelectItem key={option} value={option}>
                      {option}
                    </SelectItem>
                  ))}
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
              placeholder="Describe the main focus of your analysis"
              rows={3}
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Business Objective</label>
            <Textarea
              value={formData.business_objective}
              onChange={(e) => setFormData(prev => ({ ...prev, business_objective: e.target.value }))}
              placeholder="Describe your primary business objective"
              rows={3}
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Competitive Landscape</label>
            <Textarea
              value={formData.competitive_landscape}
              onChange={(e) => setFormData(prev => ({ ...prev, competitive_landscape: e.target.value }))}
              placeholder="Describe the competitive environment"
              rows={3}
            />
          </div>
        </CardContent>
      </Card>

      {/* Market Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            Market Configuration
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Target Markets</label>
              <MultiSelect
                options={marketSegmentOptions}
                value={formData.market_segments}
                onChange={(value) => setFormData(prev => ({ ...prev, market_segments: value }))}
                placeholder="Select target market segments"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Product Categories</label>
              <MultiSelect
                options={productCategoryOptions}
                value={formData.product_categories}
                onChange={(value) => setFormData(prev => ({ ...prev, product_categories: value }))}
                placeholder="Select relevant product categories"
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
            <label className="text-sm font-medium">Brands to Analyze</label>
            <MultiSelect
              options={brandOptions}
              value={formData.brands}
              onChange={(value) => setFormData(prev => ({ ...prev, brands: value }))}
              placeholder="Select brands for analysis"
            />
          </div>
        </CardContent>
      </Card>

      {/* Forecasting & Metrics */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Forecasting & Metrics
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Forecast Periods (days)</label>
              <Input
                type="number"
                value={formData.forecast_periods}
                onChange={(e) => setFormData(prev => ({ ...prev, forecast_periods: parseInt(e.target.value) || 30 }))}
                min={7}
                max={365}
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Expected Revenue ($)</label>
              <Input
                type="number"
                value={formData.expected_revenue}
                onChange={(e) => setFormData(prev => ({ ...prev, expected_revenue: parseInt(e.target.value) || 0 }))}
                min={1000}
                max={10000000}
                step={1000}
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
                  {marketPositionOptions.map((option) => (
                    <SelectItem key={option} value={option}>
                      {option}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Context Strategy</label>
              <Select 
                value={formData.context_strategy} 
                onValueChange={(value) => setFormData(prev => ({ ...prev, context_strategy: value }))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {contextStrategyOptions.map((option) => (
                    <SelectItem key={option} value={option}>
                      {option}
                    </SelectItem>
                  ))}
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
              <label className="text-sm font-medium">Sentiment Score</label>
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

            <div className="space-y-2">
              <label className="text-sm font-medium">Token Budget</label>
              <div className="space-y-2">
                <Slider
                  value={[formData.token_budget]}
                  onValueChange={(value) => setFormData(prev => ({ ...prev, token_budget: value[0] }))}
                  max={50000}
                  min={1000}
                  step={500}
                />
                <div className="text-sm text-gray-500">{formData.token_budget.toLocaleString()} tokens</div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
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
              <label htmlFor="enable_mem0" className="text-sm">Enable Mem0 Memory</label>
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

            <div className="flex items-center space-x-2">
              <Checkbox
                id="show_comparison"
                checked={formData.show_comparison}
                onCheckedChange={(checked) => setFormData(prev => ({ ...prev, show_comparison: !!checked }))}
              />
              <label htmlFor="show_comparison" className="text-sm">Show Performance Comparison</label>
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