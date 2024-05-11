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

* add

> conda config --add channels [URL]
>
> PS: "URL" ummary can be found in Appendix 2 at the end of the article

* clean cache

> conda clean -i

* generate configuration file

> conda config --set show_channel_urls yes
>
> The configuration file template ( ".condarc", 该文件位于用户目录当中):  
>
> > channels:  
> > \- defaults  
> > show_channel_urls: true  
> >
> > ( channel_alias: [URL] )  
> >
> > default_channels:  
> > \- [URL]  
> > ...  
> >
> > custom_channels:  
> > [type-name:URL]  
> > ...  
>
> PS: File examples can be found in Appendix 1 at the end of the article
>

* remove
* reset

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

## Appendix

### 1

* Tsinghua University Open Source Software Mirror Station

> channels:  
> \- defaults  
>
> show_channel_urls: true  
>
> channel_alias: https://mirrors.tuna.tsinghua.edu.cn/anaconda  
>
> default_channels:  
> \- https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main  
> \- https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free  
> \- https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r  
> \- https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro  
> \- https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2  
>
> custom_channels:  
> conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud  
> msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud  
> bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud  
> menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud  
> pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud  
> simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud  

* Beijing Foreign Studies University Open Source Software Mirror Station  

> channels:  
> \- defaults  
>
> show_channel_urls: true  
>
> channel_alias: https://mirrors.bfsu.edu.cn/anaconda  
>
> default_channels:  
> \- https://mirrors.bfsu.edu.cn/anaconda/pkgs/main  
> \- https://mirrors.bfsu.edu.cn/anaconda/pkgs/free  
> \- https://mirrors.bfsu.edu.cn/anaconda/pkgs/r  
> \- https://mirrors.bfsu.edu.cn/anaconda/pkgs/pro  
> \- https://mirrors.bfsu.edu.cn/anaconda/pkgs/msys2  
>
> custom_channels:  
> conda-forge: https://mirrors.bfsu.edu.cn/anaconda/cloud  
> msys2: https://mirrors.bfsu.edu.cn/anaconda/cloud  
> bioconda: https://mirrors.bfsu.edu.cn/anaconda/cloud  
> menpo: https://mirrors.bfsu.edu.cn/anaconda/cloud  
> pytorch: https://mirrors.bfsu.edu.cn/anaconda/cloud  
> simpleitk: https://mirrors.bfsu.edu.cn/anaconda/cloud  

* Shanghai Jiao Tong University Open Source Software Mirror Station

> channels:  
> \- defaults  
>
> show_channel_urls: true  
>
> channel_alias: https://anaconda.mirrors.sjtug.sjtu.edu.cn/  
>
> default_channels:  
> \- https://anaconda.mirrors.sjtug.sjtu.edu.cn/pkgs/main  
> \- https://anaconda.mirrors.sjtug.sjtu.edu.cn/pkgs/free  
> \- https://anaconda.mirrors.sjtug.sjtu.edu.cn/pkgs/mro  
> \- https://anaconda.mirrors.sjtug.sjtu.edu.cn/pkgs/msys2  
> \- https://anaconda.mirrors.sjtug.sjtu.edu.cn/pkgs/pro  
> \- https://anaconda.mirrors.sjtug.sjtu.edu.cn/pkgs/r  
>
> custom_channels:  
> conda-forge: https://anaconda.mirrors.sjtug.sjtu.edu.cn/conda-forge  
> soumith: https://anaconda.mirrors.sjtug.sjtu.edu.cn/cloud/soumith  
> bioconda: https://anaconda.mirrors.sjtug.sjtu.edu.cn/cloud/bioconda  
> menpo: https://anaconda.mirrors.sjtug.sjtu.edu.cn/cloud/menpo  
> viscid-hub: https://anaconda.mirrors.sjtug.sjtu.edu.cn/cloud/viscid-hub  
> atztogo: https://anaconda.mirrors.sjtug.sjtu.edu.cn/cloud/atztogo  

* Alibaba Open Source Software Mirror Station

> channels:  
> \- defaults  
>
> show_channel_urls: true  
>
> default_channels:  
> \- http://mirrors.aliyun.com/anaconda/pkgs/main  
> \- http://mirrors.aliyun.com/anaconda/pkgs/r  
> \- http://mirrors.aliyun.com/anaconda/pkgs/msys2  
>
> custom_channels:  
> conda-forge: http://mirrors.aliyun.com/anaconda/cloud  
> msys2: http://mirrors.aliyun.com/anaconda/cloud  
> bioconda: http://mirrors.aliyun.com/anaconda/cloud  
> menpo: http://mirrors.aliyun.com/anaconda/cloud  
> pytorch: http://mirrors.aliyun.com/anaconda/cloud  
> simpleitk: http://mirrors.aliyun.com/anaconda/cloud  

### 2

* Tsinghua University Open Source Software Mirror Station

> https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main  
> https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free  
> https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r  
> https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro  
> https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2  
> https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge  
> https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2  
> https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda  
> https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/menpo  
> https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch  
> https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/simpleitk  
