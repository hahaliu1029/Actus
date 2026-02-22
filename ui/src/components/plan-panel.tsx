'use client'

import {cn} from '@/lib/utils'
import {useState} from 'react'
import {Check, ChevronDown, ChevronUp, Clock} from 'lucide-react'
import {Button} from '@/components/ui/button'

interface PlanPanelProps {
  className?: string
}

export function PlanPanel({className}: PlanPanelProps) {
  // 1.定义组件状态&函数
  const [isExpanded, setIsExpanded] = useState(false)
  const togglePanel = () => setIsExpanded(!isExpanded)
  const steps = [
    {
      id: 1,
      status: 'completed',
      description: '使用搜索工具查找第四季度（假设为当前或最近一年的10月至12月）国内人工智能的最新新闻和发展动态，使用中文关键词如\'第四季度 AI 发展 新闻 中国\'。',
    },
    {
      id: 2,
      status: 'running',
      description: '使用搜索工具查找第四季度国际人工智能的最新新闻和发展动态，使用英文关键词如\'Q4 AI developments 2023 international\'或类似，以覆盖全球范围。',
    },
    {
      id: 3,
      status: 'running',
      description: '从搜索结果中选择多个相关URL，并使用浏览器工具访问这些页面，提取页面内容以获取详细信息，包括新闻标题、日期和来源。',
    },
  ]

  return (
    <div className={cn('bg-white rounded-xl border', className)}>
      {/* 折叠状态 */}
      {!isExpanded && <div
        className="flex flex-row items-start justify-between pr-3 relative clickable cursor-pointer rounded-xl z-99"
        onClick={togglePanel}
      >
        {/* 左侧的最新计划 */}
        <div className="flex-1 min-w-0 relative overflow-hidden">
          <div className="w-full h-9">
            <div className="flex items-center justify-center gap-2.5 w-full px-4 py-2 truncate text-gray-500">
              <Clock size={16}/>
              <div className="flex flex-col w-full gap-0.5 truncate">
                <div className="text-sm truncate">
                  使用搜索工具查找第四季度（假设为当前或最近一年的10月至12月）国内人工智能的最新新闻和发展动态，例如 xxx 等
                </div>
              </div>
            </div>
          </div>
        </div>
        {/* 右侧操作按钮&步骤信息 */}
        <div
          className="flex h-full justify-center gap-2 flex-shrink-0 items-center py-2.5"
        >
          <span className="text-xs text-gray-500">1 / 5</span>
          <ChevronUp className="text-gray-700" size={16}/>
        </div>
      </div>}
      {/* 展开状态 */}
      {isExpanded && <div
        className="flex flex-col py-4 rounded-xl z-99"
      >
        {/* 顶部留白+按钮 */}
        <div className="flex px-4 mb-4 w-full">
          <div className="flex items-start ml-auto">
            <div className="flex items-center justify-center gap-2">
              <Button onClick={togglePanel} variant="ghost" size="icon-xs" className="cursor-pointer">
                <ChevronDown className="text-gray-500" size={16}/>
              </Button>
            </div>
          </div>
        </div>
        {/* 底部的计划列表 */}
        <div className="px-4">
          <div className="bg-gray-50 rounded-lg px-2 py-3">
            {/* 任务进度信息 */}
            <div className="flex justify-between w-full px-4">
              <span className="text-gray-700 font-bold">任务进度</span>
              <div className="flex items-center gap-3">
                <span className="text-xs text-gray-500">1 / 5</span>
              </div>
            </div>
            {/* 任务列表 */}
            <div className="max-h[min(cacl(100vh-360px), 400px)] overflow-y-auto">
              {steps.map(step => (
                <div
                  key={step.id}
                  className="flex items-center text-gray-500 text-sm gap-2.5 w-full px-4 py-2 truncate"
                >
                  {/* 图标 */}
                  {
                    step.status === 'completed' ?
                      <Check size={16} className="relative t-0.5 flex-shrink-0"/> :
                      <Clock size={16} className="relative t-0.5 flex-shrink-0"/>
                  }
                  {/* 任务描述 */}
                  <div className="flex flex-col w-full truncate">
                    <div className="text-sm truncate">{step.description}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>}
    </div>
  )
}