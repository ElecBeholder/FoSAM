# GitHub仓库和Pages设置指南

本文档提供了将FoSAM项目发布到GitHub并启用GitHub Pages的步骤。

## 1. 创建GitHub仓库

1. 登录您的GitHub账户
2. 点击右上角的"+"图标，然后选择"New repository"
3. 输入仓库名称"FoSAM"
4. 添加描述："Foveated Segment Anything Model - NeurIPS 2025"
5. 选择"Public"可见性
6. 不要初始化仓库（不要添加README、.gitignore或license）
7. 点击"Create repository"

## 2. 初始化本地仓库并推送到GitHub

```bash
# 导航到项目目录
cd /home/wang/FoSAM

# 初始化Git仓库
git init

# 添加所有文件
git add .

# 提交更改
git commit -m "Initial commit"

# 添加远程仓库
git remote add origin https://github.com/YourUsername/FoSAM.git

# 推送到远程仓库的main分支
git push -u origin main
```

请将上面的`YourUsername`替换为您的GitHub用户名。

## 3. 设置GitHub Pages

1. 在GitHub上导航到您的仓库
2. 点击"Settings"选项卡
3. 在左侧菜单中滚动到"Pages"部分
4. 在"Source"下，选择"main"分支
5. 点击"Save"
6. 等待几分钟，您的GitHub Pages站点将被激活
7. 站点地址将显示在Pages设置页面顶部，通常是`https://YourUsername.github.io/FoSAM/`

## 4. 更新资源路径（如果需要）

如果您需要更新资源路径（如视频或图像），请确保在HTML文件中使用相对路径，例如：

```html
<img src="fosam_fig.pdf">
<video src="assets/demo.mp4" controls></video>
```

## 5. 自定义域名（可选）

如果您想使用自定义域名：

1. 在GitHub仓库的"Settings" -> "Pages"部分
2. 在"Custom domain"字段输入您的域名
3. 点击"Save"
4. 按照GitHub提供的指示在您的DNS提供商那里设置DNS记录

## 注意事项

- 默认的GitHub Pages站点使用Jekyll处理静态文件，如果您的项目中有下划线开头的文件或目录，它们可能会被忽略
- 每次推送到main分支后，GitHub Pages站点将自动更新，但可能需要几分钟才能看到更改
- 如果您的页面没有正确显示，请检查GitHub Actions选项卡以查看构建日志

## 额外资源

- [GitHub Pages 文档](https://docs.github.com/en/pages)
- [使用自定义域名](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site)
- [Jekyll 文档](https://jekyllrb.com/docs/) (如果您想使用Jekyll扩展功能) 