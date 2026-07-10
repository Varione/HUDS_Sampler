# HUDS Sampler 改进发现与洞察

## 代码探索发现

### 项目结构
- `huds_app/` - 主应用包
  - `core/` - 核心功能 (config, metrics, storage)
  - `data/` - 数据处理 (pool, schema, validation)
  - `interface/` - 接口层 (cli, maxwell, workflow)
  - `model/` - 模型层 (architecture, train)
  - `sampling/` - 采样算法 (huds)

### 关键文件状态
需要进一步分析每个文件的当前实现。

## 待分析问题

### P0 相关
1. CLI entry point 配置问题
2. storage.py 原子状态管理缺失
3. 运行目录锁缺失
4. MC Dropout 机制是否正确启用
5. repeat_times 验证逻辑
6. 图像输出尺寸校验缺失
7. checkpoint 加载策略
8. smoke test 缺失

### P1 相关
9. MC 均值方差计算方式
10. k-center 距离更新效率
11. 范围限制机制
12. 不确定性归一化方法
13. 采样诊断字段定义

### P2 相关
14. 配置 schema 严格性
15. 变量类型支持
16. QMC/Sobol/LHS 实现
17. 输出 tensor 结构
18. checkpoint 元数据完整性
19. CI/文档/许可证

## 技术决策记录
待补充...
