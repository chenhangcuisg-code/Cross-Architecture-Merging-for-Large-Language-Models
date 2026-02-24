# GitHub 上传指南

## 准备工作

当前代码已经准备好上传到 GitHub。仓库应命名为：
**Cross-Architecture Merging for Large Language Models**

## 上传步骤

### 方法 1: 使用提供的脚本（推荐）

在仓库根目录下执行（请先设置环境变量，不要将 token 写入脚本）：

```bash
cd /path/to/Cross-Architecture-Merging-for-Large-Language-Models   # 进入本仓库根目录
GITHUB_USERNAME=你的GitHub用户名 GITHUB_TOKEN=你的token bash setup_github.sh
```

然后按照脚本输出的提示操作。

### 方法 2: 手动操作

1. **初始化 Git 仓库**（如果还没有）：
```bash
cd /path/to/Cross-Architecture-Merging-for-Large-Language-Models
git init
```

2. **添加所有文件**：
```bash
git add .
```

3. **创建初始提交**：
```bash
git commit -m "Initial commit: Cross-Architecture Merging for Large Language Models"
```

4. **在 GitHub 上创建新仓库**：
   - 访问 https://github.com/new
   - 仓库名称：`Cross-Architecture-Merging-for-Large-Language-Models`
   - 描述：`Cross-Architecture Merging for Large Language Models`
   - 选择 Public 或 Private
   - **不要**初始化 README、.gitignore 或 license（我们已经有了）

5. **添加远程仓库并推送**（将 `YOUR_USERNAME` 替换为你的 GitHub 用户名）：
```bash
git remote add origin https://github.com/YOUR_USERNAME/Cross-Architecture-Merging-for-Large-Language-Models.git
git branch -M main
git push -u origin main
```

## 注意事项

- ✅ 代码已匿名化（已移除用户名和硬编码路径）
- ✅ 已创建 `.gitignore` 文件（排除大文件和临时文件）
- ✅ README.md 已更新为正确的标题和结构
- ⚠️ 如果需要，可以添加 LICENSE 文件

## 推送（需在本机执行）

若在无交互环境提交后尚未推送，请在本机进入仓库根目录执行（将 `YOUR_USERNAME` 和 `YOUR_TOKEN` 替换为你的 GitHub 用户名与 Personal Access Token）：

```bash
cd /path/to/Cross-Architecture-Merging-for-Large-Language-Models
git push https://YOUR_USERNAME:YOUR_TOKEN@github.com/YOUR_USERNAME/Cross-Architecture-Merging-for-Large-Language-Models.git main
```

或先配置 remote 再推送（Token 仅用于本次，不要提交到仓库）：

```bash
git remote set-url origin https://YOUR_USERNAME:YOUR_TOKEN@github.com/YOUR_USERNAME/Cross-Architecture-Merging-for-Large-Language-Models.git
git push -u origin main
```

## 验证

上传后，请检查：
- [ ] 所有代码文件都已上传
- [ ] README.md 显示正确
- [ ] 没有大文件（模型文件、数据文件等）被意外上传
- [ ] 代码可以正常克隆和运行
