<h1 align = "center"> Anaconda Prompt (DOS) - CMD </h1>

## Reading Notes

> 1. 本篇整理针对 Anaconda3( version 2.0.0 ~ 2.5.3 ) 的 DOS(UI) 界面  
> 2. 默认命令主体在 Windows 和 Ubuntu(Linux) 中通用，有特殊情况会额外补充说明  
> 3. 命令格式在不同 OS 环境中略有差别  
> 4. 命令格式说明中，'()' 代表可选（可省略），'[]' 代表 "指定格式字符串替换"， || 代表"多种格式选择"  
> 5. 命令格式有两种指定方式："--[command]", "-[command]" 前者替换为命令全称，后者为简称  
> 6. anaconda 命令一般以 "conda" 开头  
> 7. pip VS conda  

## Comparison Table

## SoftWare Basic Operate

* cd -- 切换文件路径  

* [ Drive-Letter: ] -- 切换硬盘分区  

* version  -- 查看 conda 的版本  

> conda --version  

## Help

## SoftWare Config

### Channel

## Environment

* activate  

> conda activate [env-name]

* backup / export

> conda env export > [env-name].yaml

* creat  

> conda create -n [env-name] python=[version-number]  
> conda env create -f [env-name].yaml

* deactivate

> conda deactivate

* remove

> conda remove -n [env-name] --all

### Version Management

### Package Management

#### Conda Operation

* install

> conda install [package-name]  
> conda install [package-name]=[version-number]  
> conda install [package-name] --channel [URL]  
> conda install -f [requirements-filename].txt

* search

> conda search [package-name]  

* uninstall

> conda uninstall [package-name]

* version

* update

> conda update [package-name]

#### pip Operation

* install / update

> pip install [package-name]  
> pip install -U [package-name]  
> pip install -i [URL]  
> pip install -f [requirements-filename].txt

* uninstall

> pip install [package-name]

* version

> pip -V
