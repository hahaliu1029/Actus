'use client'

import {cn} from '@/lib/utils'
import {Item, ItemActions, ItemContent, ItemDescription, ItemMedia, ItemTitle} from '@/components/ui/item'
import {Avatar, AvatarGroupCount} from '@/components/ui/avatar'
import {Eye, FileSearch, FileText} from 'lucide-react'
import {Button} from '@/components/ui/button'

interface AttachmentsMessageProps {
  className?: string
  role: string
}

export function AttachmentsMessage({className, role}: AttachmentsMessageProps) {
  const files = [
    {'id': 1, 'extension': 'pdf', 'filename': 'go+java.pdf', 'size': '2.52MB'},
    {'id': 2, 'extension': 'png', 'filename': '全家福.png', 'size': '2.52MB'},
    {'id': 3, 'extension': 'docx', 'filename': '2025年年中汇报.docx', 'size': '2.52MB'},
    {'id': 4, 'extension': 'xlsx', 'filename': '数据分析可视化看板.xsx', 'size': '2.52MB'},
    {'id': 5, 'extension': 'gif', 'filename': '数据看板动态演示.gif', 'size': '2.52MB'},
    {'id': 6, 'extension': 'py', 'filename': 'ReActAgent.py', 'size': '2.52MB'},
  ]

  // 1.判断角色是否为user，如果实则渲染用户附件列表
  if (role === 'user') {
    return (
      <div className={cn('flex flex-col flex-wrap gap-2 items-end justify-end', className)}>
        <div className="flex gap-2 flex-wrap max-w-[568px] justify-end">
          {files.map(file => (
            <Item
              key={file.id}
              variant="outline"
              className="w-[280px] bg-white p-2 flex-shrink-0 gap-2"
            >
              {/* 左侧文件图标 */}
              <ItemMedia>
                <Avatar className="size-8">
                  <AvatarGroupCount>
                    <FileText/>
                  </AvatarGroupCount>
                </Avatar>
              </ItemMedia>
              {/* 文件信息 */}
              <ItemContent className="gap-0">
                <ItemTitle className="text-sm text-gray-700">{file.filename}</ItemTitle>
                <ItemDescription className="text-xs">{file.extension} · {file.size}</ItemDescription>
              </ItemContent>
              <ItemActions>
                <Button variant="ghost" size="icon-xs" className="cursor-pointer">
                  <Eye/>
                </Button>
              </ItemActions>
            </Item>
          ))}
        </div>
      </div>
    )
  } else if (role === 'assistant') {
    // 2.渲染AI附件列表
    return (
      <div className={cn('flex flex-col flex-wrap gap-2 justify-start', className)}>
        <div className="flex gap-2 flex-wrap max-w-[568px]">
          {files.map(file => (
            <Item
              key={file.id}
              variant="outline"
              className="w-[280px] bg-white p-2 flex-shrink-0 gap-2"
            >
              {/* 左侧文件图标 */}
              <ItemMedia>
                <Avatar className="size-8">
                  <AvatarGroupCount>
                    <FileText/>
                  </AvatarGroupCount>
                </Avatar>
              </ItemMedia>
              {/* 文件信息 */}
              <ItemContent className="gap-0">
                <ItemTitle className="text-sm text-gray-700">{file.filename}</ItemTitle>
                <ItemDescription className="text-xs">{file.extension} · {file.size}</ItemDescription>
              </ItemContent>
              <ItemActions>
                <Button variant="ghost" size="icon-xs" className="cursor-pointer">
                  <Eye/>
                </Button>
              </ItemActions>
            </Item>
          ))}
          <Button variant="outline" className="cursor-pointer">
            <FileSearch size={16}/>
            <span className="text-sm text-gray-700">查看此任务中所有的文件</span>
          </Button>
        </div>
      </div>
    )
  }
}