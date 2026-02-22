'use client'

import {cn} from '@/lib/utils'
import {CheckIcon, ChevronDown, Languages} from 'lucide-react'
import {ManusIcon} from '@/components/manus-icon'
import {ToolUse} from '@/components/tool-use'
import {Button} from '@/components/ui/button'
import {AttachmentsMessage} from '@/components/attachments-message'

interface ChatMessageProps {
  className?: string
  message: {
    type: string
    role?: string
  }
}

export function ChatMessage({className, message}: ChatMessageProps) {
  // 1.消息为user时显示组件
  if (message.type === 'user') {
    return (
      <div className={cn('flex w-full flex-col items-end justify-end gap-1 group mt-3', className)}>
        {/* 顶部时间 */}
        <div className="flex items-end">
          <div className="flex items-center justify-end gap-1 invisible group-hover:visible">
            <div className="float-right transition text-xs text-muted-foreground invisible group-hover:visible">
              2个月前
            </div>
          </div>
        </div>
        {/* 底部用户消息 */}
        <div className="flex max-w-[90%] relative flex-col gap-2 items-end">
          <div className="text-foreground/85 relative flex items-center rounded-lg overflow-hidden bg-card p-3 border border-border">
            帮我写一个Python版本的冒泡排序
          </div>
        </div>
      </div>
    )
  } else if (message.type === 'assistant') {
    // 2.消息为AI时显示组件
    return (
      <div className={cn('flex flex-col gap-2 w-full group mt-3', className)}>
        {/* AI图标&时间 */}
        <div className="flex items-center justify-between h-7 group">
          <div className="flex items-center justify-center gap-1 text-foreground/85">
            <Languages size={18}/>
            <ManusIcon/>
          </div>
          <div className="flex items-center gap-[3px] invisible group-hover:visible">
            <div className="float-right transition text-xs text-muted-foreground invisible group-hover:visible">
              2个月前
            </div>
          </div>
        </div>
        {/* AI消息 */}
        <div className="max-w-none p-0 m-0 text-foreground/85">
          用户请求编写一个Python版本的冒泡排序算法。冒泡排序是一种简单的排序算法，通过重复遍历列表，比较相邻元素并交换它们的位置，直到列表完全排序。我将创建一个Python脚本来实现这个算法，包括必要的注释和示例使用。这个任务需要编写代码并保存到文件中，以便后续执行或修改。
        </div>
      </div>
    )
  } else if (message.type === 'tool') {
    // 3.消息为工具时显示组件
    return (
      <ToolUse/>
    )
  } else if (message.type === 'step') {
    // 4.消息为子步骤时显示组件
    return (
      <div className="flex flex-col">
        {/* 步骤描述 */}
        <div className="text-sm w-full clickable flex gap-2 justify-between group/header truncate text-foreground/85">
          <div className="flex flex-row gap-2 justify-center items-center truncate">
            {/* 已完成状态/未完成状态 */}
            <div className="w-4 h-4 flex-shrink-0 flex items-center justify-center border rounded-[15px] bg-muted-foreground/50">
              <CheckIcon className="text-white" size={10}/>
            </div>
            {/* 步骤描述 */}
            <div className="truncate font-medium markdown-content">
              编写一个Golang程序文件，实现冒泡排序算法，包括必要的函数和主函数
            </div>
            {/* 展开or折叠icon */}
            <Button variant="ghost" size="icon-xs">
              <ChevronDown/>
            </Button>
          </div>
        </div>
        {/* 步骤详情 */}
        <div className="flex">
          <div className="w-6 relative">
            <div
              className="h-[calc(100%+14px)] border-l border-dashed border-absolute start-[8px] top-0 bottom-0"
            ></div>
          </div>
          {/* 调用工具列表信息 */}
          <div
            className="flex flex-col gap-3 flex-1 min-w-0 overflow-hidden pt-2 transition-[max-height,opacity] duration-150 ease-in-out">
            {
              [1, 2, 3, 4].map(item => (
                <ToolUse key={item}/>
              ))
            }
          </div>
        </div>
      </div>
    )
  } else if (message.type === 'attachments') {
    return (
      <AttachmentsMessage role={message.role ?? 'user'}/>
    )
  }
}