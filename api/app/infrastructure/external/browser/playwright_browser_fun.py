GET_VISIBLE_CONTENT_FUNC = """() => {
    // 1.定义变量存储所有可视元素+去重key+视口宽高
    const visibleElements = [];
    const seenContentKeys = new Set();
    const viewportHeight = window.innerHeight;
    const viewportWidth = window.innerWidth;
    const MAX_TEXT_LENGTH = 300;

    // 2.定义文本处理函数
    const normalizeText = (value) => String(value || '').replace(/\\s+/g, ' ').trim();
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

        visibleElements.push(`<${tagName}>${escapeHtml(truncatedText)}</${tagName}>`);
    }

    // 11.将所有内容使用空格拼接后包裹在div内返回
    return '<div>' + visibleElements.join(' ') + '</div>'
}"""

# 在执行代码前先执行这段js代码，实现将console.log内存存储到window.console.logs中
INJECT_CONSOLE_LOGS_FUNC = """() => {
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
}"""
