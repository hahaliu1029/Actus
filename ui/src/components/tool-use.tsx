'use client'

import {SquareChevronRight} from 'lucide-react'

export function ToolUse() {
  return (
    <>
      <p className="text-muted-foreground text-sm overflow-hidden text-ellipsis whitespace-pre-line">
        我将使用文件写入工具来创建一个包含冒泡排序算法的Golang程序文件。
      </p>
      <div className="flex items-center group gap-2 cursor-pointer">
        {/* 左侧工具信息 */}
        <div className="flex-1 min-w-0">
          <div
            className="rounded-[15px] inline-flex items-center gap-2 px-[10px] py-[3px] border border-border bg-muted max-w-full clickable"
          >
            {/* 图标信息 */}
            <div className="w-4 inline-flex items-center text-foreground/85">
              <SquareChevronRight size={21}/>
            </div>
            {/* 工具信息 */}
            <div className="flex-1 h-full min-w-0 flex">
              <div className="inline-flex items-center h-full rounded-full text-xs text-foreground/85 max-w-full">
                <div className="flex items-center justify-center">
                  正在写入文件
                  <span
                    className="flex-1 min-w-0 rounded-[6px] px-1 ml-1 relative top-0 text-xs font-mono max-w-full txt-ellipsis overflow-hidden whitespace-nowrap text-muted-foreground"
                  >
                    <code>bubble_sort.go</code>
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
        {/* 右侧时间 */}
        <div className="float-right transition text-xs text-muted-foreground invisible group-hover:visible">
          2个月前
        </div>
      </div>
    </>
  )
}