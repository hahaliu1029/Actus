/*getVisibleContent*/
const getVisibleContent = () => {
    // 1.定义变量存储所有可视元素+去重key+视口宽高
    const visibleElements = [];
    const seenContentKeys = new Set();
    const viewportHeight = window.innerHeight;
    const viewportWidth = window.innerWidth;
    const MAX_TEXT_LENGTH = 300;

    // 2.定义文本处理函数
    const normalizeText = (value) => String(value || '').replace(/\s+/g, ' ').trim();
    const escapeHtml = (value) => String(value || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    const truncateText = (value) =>
        value.length > MAX_TEXT_LENGTH
            ? value.substring(0, MAX_TEXT_LENGTH - 3) + '...'
            : value;

    // 3.获取页面上所有元素
    const elements = document.querySelectorAll("body *");

    // 4.循环遍历所有元素逐个处理
    for (let i = 0; i < elements.length; i++) {
        // 5.获取元素的尺寸与位置
        const element = elements[i];
        const rect = element.getBoundingClientRect();

        // 6.判断元素的宽高，如果没有大小则跳过
        if (rect.height === 0 || rect.width === 0) continue;

        // 7.排除完全在当前屏幕可视区域之外的元素(上方、下方、左侧、右侧)的元素
        if (
            rect.bottom < 0 ||
            rect.top > viewportHeight ||
            rect.right < 0 ||
            rect.left > viewportWidth
        ) continue;

        // 8.通用样式判断当前元素是否隐藏
        const style = window.getComputedStyle(element);
        if (
            style.display === 'none' || // 块隐藏
            style.visibility === 'hidden' || // 隐藏不可见
            style.opacity === '0' // 透明度为0
        ) continue;

        // 9.提取文本并构建去重key
        const tagName = element.tagName.toLowerCase();
        const innerText = normalizeText(element.innerText);
        const ariaLabel = normalizeText(element.getAttribute('aria-label'));
        const title = normalizeText(element.getAttribute('title'));
        const alt = normalizeText(element.getAttribute('alt'));
        const placeholder = normalizeText(element.getAttribute('placeholder'));
        const value = normalizeText(element.value);
        const href = normalizeText(element.getAttribute('href'));
        const src = normalizeText(element.getAttribute('src'));
        const inputType = normalizeText(element.getAttribute('type'));

        const isInteractiveOrMedia =
            tagName === "img" ||
            tagName === "input" ||
            tagName === "button" ||
            tagName === "textarea" ||
            tagName === "select" ||
            tagName === "a";

        let contentText = innerText;
        if (!contentText && isInteractiveOrMedia) {
            contentText =
                ariaLabel ||
                alt ||
                title ||
                placeholder ||
                value ||
                "[No text]";
        }

        if (!contentText) continue;

        const truncatedText = truncateText(contentText);
        let contentKey = `text:${truncatedText}`;
        if (isInteractiveOrMedia) {
            contentKey = `node:${tagName}|${inputType}|${href}|${src}|${truncatedText}`;
        }

        // 10.按首见顺序去重，减少重复信息和token
        if (seenContentKeys.has(contentKey)) continue;
        seenContentKeys.add(contentKey);

        visibleElements.push(`<${tagName}>${escapeHtml(truncatedText)}</${tagName}>`)
    }

    // 11.将所有内容使用空格拼接后包裹在div内返回
    return '<div>' + visibleElements.join(' ') + '</div>'
}
/*getVisibleContent*/

/*getInteractiveElements*/
const getInteractiveElements = () => {
    // 1.定义变量存储激活元素列表+视口宽高
    const interactiveElements = [];
    const viewportHeight = window.innerHeight;
    const viewportWidth = window.innerWidth;

    // 2.获取页面上所有可交互的元素，包含原生可交互标签 + 常见ARIA交互角色
    const selectorParts = [
        'button',
        'a',
        'input',
        'textarea',
        'select',
        '[role="button"]',
        '[role="link"]',
        '[role="menuitem"]',
        '[role="menuitemcheckbox"]',
        '[role="menuitemradio"]',
        '[role="tab"]',
        '[role="option"]',
        '[role="checkbox"]',
        '[role="radio"]',
        '[role="switch"]',
        '[role="textbox"]',
        '[role="searchbox"]',
        '[role="combobox"]',
        '[role="slider"]',
        '[role="spinbutton"]',
        '[tabindex]:not([tabindex="-1"])'
    ];
    const elements = document.querySelectorAll(selectorParts.join(', '));

    // 3.定义变量用于生成连续的唯一索引
    let validElementIndex = 0;

    // 4.循环遍历所有元素
    for (let i = 0; i < elements.length; i++) {
        // 5.取出对应元素并获取尺寸+位置
        const element = elements[i];
        const rect = element.getBoundingClientRect();

        // 6.宽高任意为0则跳过这条元素
        if (rect.width === 0 || rect.height === 0) continue;

        // 7.视口不可见该元素则跳过
        if (
            rect.bottom < 0 ||
            rect.top > viewportHeight ||
            rect.right < 0 ||
            rect.left > viewportWidth
        ) continue;

        // 8.样式不可见则跳过该元素
        const style = window.getComputedStyle(element);
        if (
            style.display === 'none' ||
            style.visibility === 'hidden' ||
            style.opacity === '0'
        ) continue;


        // 9.获取元素的标签名并转换成小写，同时提取标签内容
        let tagName = element.tagName.toLowerCase();
        let text = '';

        // 10.根据标签类型不同处理不同的逻辑，首先是输入框/文本域/下拉菜单
        if (element.value && ['input', 'textarea', 'select'].includes(tagName)) {
            text = element.value;

            // 11.标签为输入框则执行以下代码，记录label和placeholder
            if (tagName === 'input') {
                // 12.查询输入框的label是否存在并赋值
                let labelText = '';
                if (element.id) {
                    const label = document.querySelector(`label[for="${element.id}"]`);
                    if (label) {
                        labelText = label.innerText.trim();
                    }
                }

                // 13.查找父级或同级的 label (当没有 for 属性绑定时)
                if (!labelText) {
                    const parentLabel = element.closest('label');
                    if (parentLabel) {
                        labelText = parentLabel.innerText.trim().replace(element.value, '').trim();
                    }
                }

                // 14.拼接label消息
                if (labelText) {
                    text = `[Label: ${labelText}] ${text}`;
                }

                // 15.拼接placeholder信息
                if (element.placeholder) {
                    text = `${text} [Placeholder: ${element.placeholder}]`;
                }
            }
        } else if (element.innerText) {
            // 16.普通元素则提取内部文本并剔除多余空格 (如 <button>提交</button>)
            text = element.innerText.trim().replace(/\\s+/g, ' ');
        } else if (element.getAttribute('aria-label')) {
            text = element.getAttribute('aria-label').trim();
        } else if (element.alt) {
            // 17.图片按钮，取 alt 属性
            text = element.alt;
        } else if (element.title) {
            // 18.取 title 属性
            text = element.title;
        } else if (element.placeholder) {
            // 19.提取placeholder
            text = `[Placeholder: ${element.placeholder}]`;
        } else if (element.type) {
            // 20.兜底逻辑将元素的类型作为文本描述
            text = `[${element.type}]`;

            // 21.针对没有值的 Input，再次尝试获取 Label 和 Placeholder (逻辑同上)
            if (tagName === 'input') {
                let labelText = '';
                if (element.id) {
                    const label = document.querySelector(`label[for="${element.id}"]`);
                    if (label) {
                        labelText = label.innerText.trim();
                    }
                }

                if (!labelText) {
                    const parentLabel = element.closest('label');
                    if (parentLabel) {
                        labelText = parentLabel.innerText.trim();
                    }
                }

                if (labelText) {
                    text = `[Label: ${labelText}] ${text}`;
                }

                if (element.placeholder) {
                    text = `${text} [Placeholder: ${element.placeholder}]`;
                }
            }
        } else {
            // 22.都不满足，则设置为No text
            text = '[No text]';
        }

        // 23.检测文本长度是否超过100，如果是则剔除多余的部分
        if (text.length > 100) {
            text = text.substring(0, 97) + '...';
        }

        // 24.为当前元素添加data-manus-id的属性，值为manus-element-idx，这样可以通过索引找到对应的元素
        element.setAttribute('data-manus-id', `manus-element-${validElementIndex}`);

        // 25.构建css选择器
        const selector = `[data-manus-id="manus-element-${validElementIndex}"]`;

        // 26.将索引、标签名、文本、选择器添加到激活元素列表中
        interactiveElements.push({
            index: validElementIndex,
            tag: tagName,
            text: text,
            selector: selector
        });

        // 27.索引自增1
        validElementIndex++;
    }

    // 28.最终返回所有激活元素数据
    return interactiveElements;
}
/*getInteractiveElements*/

/*injectConsoleLogs***/
const injectConsoleLogs = () => {
    const MAX_LOGS = 1000;
    const levels = ['log', 'info', 'warn', 'error', 'debug'];

    const ensureLogContainer = () => {
        if (!Array.isArray(window.console.logs)) {
            window.console.logs = [];
        }
    };

    const stringifyArg = (arg) => {
        if (typeof arg === 'string') {
            return arg;
        }
        if (arg instanceof Error) {
            return arg.stack || arg.message;
        }
        try {
            return JSON.stringify(arg);
        } catch (_error) {
            return String(arg);
        }
    };

    const pushLog = (level, args) => {
        ensureLogContainer();
        const line = `[${level.toUpperCase()}] ` + args.map(stringifyArg).join(' ');
        window.console.logs.push(line);
        if (window.console.logs.length > MAX_LOGS) {
            window.console.logs.splice(0, window.console.logs.length - MAX_LOGS);
        }
    };

    ensureLogContainer();

    if (window.__manusConsoleHooked) {
        return true;
    }

    window.__manusConsoleHooked = true;
    window.__manusConsoleOriginal = window.__manusConsoleOriginal || {};

    levels.forEach((level) => {
        const original = console[level];
        window.__manusConsoleOriginal[level] = original;
        if (typeof original !== 'function') {
            return;
        }

        console[level] = (...args) => {
            pushLog(level, args);
            original.apply(console, args);
        };
    });

    return true;
}
/*injectConsoleLogs***/
