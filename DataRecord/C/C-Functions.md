<h1 align = "center"> C Functions </h1>

## stdio.h

    scanf
    sscanf
    printf
    snprintf
    getc
    fgetc
    gets
    fgets

## math.h

    exp: 计算e的指数。
    ldexp: 返回x乘以2的指数次幂。
    log10: 返回x的以10为底的对数。
    log: 返回x的自然对数。
    fmod: 返回x除以y的余数。
    pow: 返回x的y次幂。
    sqrt: 返回x的平方根。
    ceil: 返回不小于x的最小整数。
    floor: 返回不大于x的最大整数。
    fabs, fabsf, fabsl: 返回x的绝对值。

## string.h

    memchr: 在内存块中查找第一次出现的字符位置。
    memcmp: 比较两个内存块的内容。
    memcpy: 复制内存块的内容。
    memmove: 移动内存块的内容，可用于重叠区域。
    memset: 设置内存块中的所有字节为特定值。
    strcat: 连接两个字符串。
    strncat: 连接字符串，但只添加固定数量的字符。
    strchr: 查找字符串中第一次出现的字符位置。
    strrchr: 查找字符串中最后一次出现的字符位置。
    strstr: 在字符串中查找子字符串的位置。
    strpbrk: 查找字符串中第一个出现在另一字符串中的字符位置。
    strcmp: 比较两个字符串。
    strncmp: 比较两个字符串，但只比较固定数量的字符。
    strcoll: 根据当前的排序规则比较两个字符串。

## stdlib.h

    bsearch: 在已排序的数组中查找元素。
    qsort: 对数组进行快速排序。
    atoi: 将字符串转换为整数。
    atof: 将字符串转换为浮点数。
    atol: 将字符串转换为长整数。
    atoll: 将字符串转换为长长整数。
    strtol: 将字符串转换为长整数。
    strtoll: 将字符串转换为长长整数。
    strtoul: 将字符串转换为无符号长整数。
    strtoull: 将字符串转换为无符号长长整数。
    rand: 生成随机数。
    srand: 设置随机数种子。
    malloc: 分配内存。
    calloc: 分配并初始化内存。
    realloc: 重新分配内存。
    free: 释放内存。
    exit: 退出程序。
    abort: 中止程序。

## time.h

    time: 返回当前日历时间。
    ctime: 返回日历时间字符串。
    asctime: 返回日历时间字符串。
    gmtime: 返回日历时间结构。
    localtime: 返回日历时间结构。
    mktime: 返回日历时间。
    strftime: 格式化日历时间字符串。
    difftime: 返回两个日历时间的差值。
    clock: 返回程序运行时间。
    clock_gettime: 返回程序运行时间。
    clock_settime: 设置程序运行时间。
