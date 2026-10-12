"""
data_generator.py - 模拟对话数据生成器

生成多主题、多日期的模拟对话，用于测试大规模检索。

设计思路：
- 5 个主题，每个主题多个模板
- 变量池足够大，保证生成 1 万条不重复
- 生成时自动去重，确保 user 消息唯一
- 时间跨度 3 个月，模拟真实聊天的时间分布

运行方式:
  python data_generator.py
"""

import sys
import os
import json
import random
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))

# 固定随机种子，保证数据可复现
random.seed(42)

THEMES = {
    "cpp_programming": {
        "name": "C++ 编程",
        "user_templates": [
            "我在用 C++ 写一个 {topic}，遇到了 {problem} 的问题。",
            "C++ 的 {feature} 怎么用？能不能举个例子。",
            "我的 C++ 代码里 {problem} 导致了内存泄漏，怎么排查？",
            "C++ {version} 的 {feature} 和旧版本有什么区别？",
            "用 C++ 实现 {pattern} 模式，有没有简洁的写法？",
            "C++ 项目里 {topic} 的性能不太行，有什么优化思路？",
            "在 C++ {version} 里用 {feature}，编译器报了 {problem}，怎么办？",
            "我想在 C++ 里实现一个线程安全的 {topic}，有什么建议？",
            "C++ 的 {feature} 和 {other_feature} 有什么区别？该选哪个？",
            "用 C++ 写 {topic} 时，{warning} 怎么处理？",
        ],
        "assistant_templates": [
            "C++ 中 {solution}。建议用 {tool} 来处理 {problem}。",
            "{feature} 的用法：{solution}。注意 {warning}。",
            "内存泄漏排查：{solution}。推荐用 Valgrind 或 AddressSanitizer。",
            "C++{version} 引入了 {feature}，{solution}，比旧方式更安全。",
            "实现 {pattern} 模式：{solution}，利用 RAII 管理资源生命周期。",
            "性能优化：{solution}，配合 {tool} 分析瓶颈。",
            "线程安全：{solution}，注意 {warning}。",
            "{feature} 和 {other_feature} 的区别：{solution}。看场景选择。",
        ],
        "vars": {
            "topic": ["智能指针", "模板元编程", "并发队列", "内存池", "序列化库", "网络框架", "协程库", "状态机",
                       "环形缓冲区", "对象池", "事件循环", "信号槽系统", "插件架构", "序列化协议", "内存映射文件",
                       "无锁队列", "线程池", "异步 IO 模型", "表达式模板", "CRTP 基类", "委托模式"],
            "problem": ["段错误", "数据竞争", "内存泄漏", "性能瓶颈", "编译错误", "链接错误", "未定义行为", "资源泄漏",
                         "死锁", "栈溢出", "浮点精度丢失", "迭代器失效", "对象切片", "悬垂指针", "双重释放",
                         "缓冲区溢出", "整数溢出", "野指针", "内存碎片", "缓存不命中"],
            "feature": ["RAII", "std::move", "constexpr", "concepts", "coroutines", "std::variant", "std::optional",
                         "智能指针", "std::span", "ranges", "std::format", "结构化绑定", "if constexpr",
                         "折叠表达式", "std::expected", "std::any", "std::function", "模板特化"],
            "other_feature": ["裸指针", "std::shared_ptr", "传统回调", "虚函数", "宏定义", "手动内存管理",
                               "std::any", "std::variant", "继承多态", "函数指针"],
            "version": ["11", "14", "17", "20", "23"],
            "tool": ["Valgrind", "gdb", "AddressSanitizer", "perf", "clang-tidy", "cppcheck",
                      "strace", "ltrace", "Intel VTune", "gprof", "bcc/eBPF", "rr debugger"],
            "warning": ["生命周期管理", "线程安全", "异常安全", "性能开销", "内存对齐",
                         "ABI 兼容性", "编译器差异", "未指定行为", "缓存行对齐", "伪共享"],
            "solution": [
                "用 std::unique_ptr 管理所有权", "使用 std::lock_guard 保护共享数据",
                "通过 move 语义避免不必要的拷贝", "用 std::atomic 替代互斥锁",
                "使用 constexpr 在编译期计算", "用 RAII 包装资源释放",
                "通过 std::variant 实现类型安全的联合体", "用 std::jthread 替代 std::thread",
                "通过 if constexpr 简化模板", "用 std::span 替代指针+长度",
                "使用 ranges 简化算法链式调用", "用 std::format 替代 printf",
                "通过 concepts 约束模板参数", "用 coroutines 简化异步代码",
            ],
            "pattern": ["观察者", "工厂", "策略", "单例", "责任链", "状态",
                         "命令", "访问者", "装饰器", "适配器", "代理", "原型",
                         "组合", "桥接", "享元"],
        }
    },
    "project_discussion": {
        "name": "项目讨论",
        "user_templates": [
            "我们的项目需要实现 {feature}，你觉得用什么方案？",
            "项目里的 {module} 模块重构了，{problem} 怎么处理？",
            "我在做 {project_type} 项目，架构上 {question}？",
            "项目进度 {status}，{task} 还没做完，优先级怎么排？",
            "客户要求 {requirement}，技术上可行吗？",
            "{project_type} 项目的 {module} 模块 {problem}，有什么经验分享？",
            "团队在讨论 {feature} 的技术选型，{option_a} 还是 {option_b}？",
            "项目上线后发现 {problem}，怎么紧急修复？",
            "我们的 {project_type} 项目要支持 {requirement}，架构怎么调？",
            "{module} 模块的 {task} 做完了，下一步该做什么？",
        ],
        "assistant_templates": [
            "建议用 {approach}，理由是 {reason}。",
            "{module} 重构要注意 {warning}，推荐 {approach}。",
            "{project_type} 项目推荐 {architecture} 架构，{reason}。",
            "优先做 {priority_task}，{reason}。{task} 可以往后放。",
            "{requirement} 技术上可行，用 {tech} 实现，成本约 {cost}。",
            "技术选型：{option_a} 适合 {reason}，{option_b} 适合 {reason}。",
        ],
        "vars": {
            "feature": ["分布式锁", "消息队列", "缓存层", "日志聚合", "配置中心", "熔断降级",
                         "限流", "链路追踪", "灰度发布", "蓝绿部署", "服务网格", "API 网关",
                         "分布式事务", "数据分片", "读写分离", "CDN 加速", "WebSocket 长连接",
                         "定时任务调度", "文件存储", "全文搜索引擎"],
            "module": ["用户认证", "支付", "订单", "搜索", "推荐", "数据分析",
                        "消息通知", "权限管理", "商品管理", "库存", "购物车", "评价系统",
                        "售后", "物流", "营销活动", "会员体系", "风控", "对账",
                        "数据同步", "报表导出"],
            "problem": ["接口不兼容", "性能下降", "测试覆盖不足", "依赖混乱",
                         "并发冲突", "数据不一致", "内存溢出", "响应超时",
                         "连接泄漏", "缓存雪崩", "消息丢失", "重复消费",
                         "死锁", "慢查询", "序列化错误", "版本回滚困难"],
            "project_type": ["电商", "SaaS", "物联网", "金融", "游戏", "教育",
                              "医疗", "物流", "社交", "内容平台", "直播", "广告",
                              "CRM", "ERP", "协同办公"],
            "question": ["微服务还是单体", "用不用 K8s", "数据库怎么选", "缓存用什么",
                          "消息队列选哪个", "要不要上 Service Mesh", "日志怎么收集",
                          "怎么做服务发现", "配置怎么管理", "网关用什么"],
            "status": ["延期了", "正常", "超前", "卡住了", "重新评估中", "等依赖方"],
            "task": ["接口联调", "性能测试", "安全审计", "文档编写",
                      "压力测试", "灰度验证", "数据迁移", "监控接入",
                      "告警配置", "日志排查"],
            "requirement": ["支持百万并发", "毫秒级响应", "跨平台部署", "实时数据分析",
                             "多租户隔离", "数据加密", "高可用 99.99%", "弹性伸缩",
                             "离线同步", "国际化", "审计日志", "数据脱敏"],
            "approach": ["Redis 分布式锁", "Kafka 异步消息", "Redis 缓存", "ELK 日志",
                          "Nacos 配置中心", "Sentinel 熔断", "SkyWalking 链路追踪",
                          "Nginx 网关", "ShardingSphere 分库分表", "Canal 数据同步",
                          "RocketMQ 延迟消息", "Eureka 服务发现"],
            "reason": ["成熟稳定", "社区活跃", "性能好", "团队熟悉",
                        "文档完善", "运维成本低", "扩展性强", "生态丰富",
                        "学习曲线平缓", "有成功案例"],
            "architecture": ["微服务", "DDD 分层", "事件驱动", "CQRS",
                              "六边形架构", "整洁架构", "SOA", "Serverless"],
            "priority_task": ["核心链路", "支付流程", "用户注册", "安全修复",
                               "数据迁移", "性能优化", "告警接入"],
            "tech": ["Go + gRPC", "Java + Spring Cloud", "Rust + tonic",
                      "Node + NestJS", "Python + FastAPI", "C++ + brpc"],
            "cost": ["2 人周", "1 人月", "3 人天", "半个月", "1 人季度", "3 人周"],
            "option_a": ["Redis", "Kafka", "MySQL", "单体架构", "自研"],
            "option_b": ["Memcached", "RabbitMQ", "PostgreSQL", "微服务", "开源方案"],
            "warning": ["数据一致性", "幂等性", "事务边界", "服务依赖",
                         "回滚方案", "灰度策略", "监控告警"],
        }
    },
    "daily_life": {
        "name": "日常生活",
        "user_templates": [
            "今天 {activity}，感觉 {feeling}。",
            "我想 {plan}，有什么建议吗？",
            "最近 {situation}，有点 {emotion}。",
            "周末打算 {weekend_plan}，你觉得怎么样？",
            "今天看到一个 {thing}，{comment}。",
            "最近在 {hobby}，{hobby_feeling}。",
            "今天 {weather}，适合 {weather_activity}。",
            "晚上 {evening_activity}，感觉 {feeling}。",
            "最近 {health}，要不要 {health_advice}？",
            "今天中午 {food}，{food_comment}。",
        ],
        "assistant_templates": [
            "听起来{feeling}！{suggestion}。",
            "{plan} 的话，{suggestion}。注意 {tip}。",
            "{situation} 确实会让人{emotion}，{suggestion}。",
            "周末 {weekend_plan} 不错，{suggestion}。",
            "哈哈，{thing} 确实{comment}。{suggestion}。",
        ],
        "vars": {
            "activity": ["去爬山", "跑了 5 公里", "学了新菜", "看了部电影", "加班到很晚",
                          "去了趟超市", "打扫了房间", "修好了坏掉的水龙头", "去游泳了",
                          "骑了 20 公里自行车", "去公园散步", "逛了书店", "做了顿火锅",
                          "去看了展览", "打了一下午游戏", "去钓鱼了", "参加了读书会",
                          "去健身房练了背", "去朋友家聚餐", "去看了日出"],
            "feeling": ["很累但充实", "心情不错", "有点疲惫", "很有收获", "特别开心",
                         "有点无聊", "精神焕发", "腰酸背痛", "意犹未尽", "放松了不少"],
            "plan": ["学做菜", "开始健身", "学一门新语言", "去旅行", "整理房间",
                      "学吉他", "养一盆植物", "学摄影", "写日记", "学游泳",
                      "考个证", "学投资理财", "做志愿者", "学冥想", "装修房子"],
            "situation": ["工作压力大", "睡眠不太好", "和同事有分歧", "项目节奏太快",
                           "搬家很麻烦", "感冒了", "加班太多", "和家人吵架了",
                           "天气太热", "堵车严重", "手机坏了", "快递丢了",
                           "失眠", "肩膀疼", "预算超支"],
            "emotion": ["焦虑", "烦躁", "低落", "紧张", "郁闷", "心累",
                         "不知所措", "想躺平", "有点慌", "脾气暴躁"],
            "weekend_plan": ["去郊外露营", "在家打游戏", "约朋友吃饭", "去图书馆",
                              "去爬山", "逛街", "看演唱会", "去博物馆",
                              "在家做饭", "去海边", "去钓鱼", "骑车环城",
                              "泡温泉", "逛花鸟市场", "去农家乐"],
            "thing": ["有趣的视频", "好听的歌", "搞笑的段子", "惊艳的照片",
                       "可爱的猫", "好吃的餐厅", "有意思的书", "漂亮的日落",
                       "好玩的app", "感动的故事", "厉害的画画", "搞笑的表情包",
                       "治愈的动画", "硬核的科普", "意想不到的新闻"],
            "comment": ["太有意思了", "很有感触", "笑死我了", "真的很美",
                         "太治愈了", "学到了", "馋了", "想去看看",
                         "太离谱了", "很有启发"],
            "hobby": ["学插花", "练书法", "学画画", "弹钢琴", "下棋",
                       "做手工", "学陶艺", "练瑜伽", "学烘焙", "养多肉",
                       "学剪辑", "写小说", "学魔术", "练口语", "学编织"],
            "hobby_feeling": ["进步挺快", "遇到瓶颈了", "越来越感兴趣", "有点想放弃", "找到感觉了"],
            "weather": ["阳光很好", "下小雨", "有点阴", "大风", "雾霾",
                         "闷热", "凉爽", "下了暴雨", "飘雪了", "刮台风"],
            "weather_activity": ["出门走走", "在家窝着", "跑步", "爬山", "拍照",
                                   "喝咖啡看书", "逛商场", "去博物馆", "在家做饭"],
            "evening_activity": ["看了个纪录片", "和朋友视频聊天", "加班赶报告",
                                   "刷了会儿短视频", "读了半本书", "出去夜跑",
                                   "去楼下散步", "练了一会儿琴", "写了点东西"],
            "health": ["颈椎不舒服", "眼睛干涩", "腰有点疼", "总是犯困", "体重涨了",
                        "嗓子不舒服", "肩膀僵硬", "睡眠质量差", "容易焦虑", "胃口不好"],
            "health_advice": ["去医院检查一下", "注意休息", "调整作息", "多做运动", "少吃外卖"],
            "food": ["吃了火锅", "点的外卖", "自己做的炒饭", "吃了日料", "吃的面条",
                      "吃了烤肉", "做了三明治", "吃了沙拉", "喝的粥", "吃了饺子",
                      "做了咖喱饭", "吃了披萨", "吃的川菜", "做了寿司", "吃了碗螺蛳粉"],
            "food_comment": ["味道不错", "一般般", "吃撑了", "太辣了", "很健康",
                              "下次换一家", "自己做的就是香", "太咸了", "分量足"],
            "suggestion": ["注意休息", "坚持就好", "慢慢来", "可以试试", "随性一点",
                            "别给自己太大压力", "先做最重要的事", "享受过程就好",
                            "适当放松一下", "保持节奏"],
            "tip": ["别太拼", "循序渐进", "做好防护", "提前预约", "注意安全",
                     "带够水", "做好防晒", "别熬夜", "多喝水", "做好预算"],
        }
    },
    "learning_notes": {
        "name": "学习笔记",
        "user_templates": [
            "我在学 {subject}，{question}？",
            "{subject} 的 {concept} 怎么理解？",
            "看了 {book} 的第 {chapter} 章，{takeaway}。",
            "{subject} 和 {other_subject} 有什么区别？",
            "学习 {subject} 时，{problem} 怎么解决？",
            "{subject} 里的 {concept} 在实际中有什么用？",
            "想入门 {subject}，有什么学习路线推荐？",
            "{book} 这本书值得看吗？{question}？",
            "面试被问到 {concept}，我答得不太好，求详细解释。",
            "{subject} 的 {concept} 和 {other_concept} 容易混淆，怎么区分？",
        ],
        "assistant_templates": [
            "{subject} 中 {answer}。{example}。",
            "{concept} 的理解：{answer}。{example}。",
            "{book} 这章讲得不错，{answer}。{example}。",
            "主要区别：{answer}。{example}。",
            "{problem} 的解决方案：{answer}。{example}。",
            "实际应用：{answer}。{example}。",
            "学习路线：先掌握 {concept}，再学 {other_concept}。",
        ],
        "vars": {
            "subject": ["操作系统", "计算机网络", "数据结构", "算法", "数据库",
                         "分布式系统", "编译原理", "机器学习", "密码学", "图形学",
                         "软件工程", "设计模式", "并发编程", "网络编程", "内核原理",
                         "容器技术", "函数式编程", "系统设计", "计算机体系结构", "信息安全"],
            "concept": ["虚拟内存", "TCP 三次握手", "红黑树", "动态规划", "B+ 树",
                         "CAP 理论", "词法分析", "反向传播", "RSA 加密", "光栅化",
                         "SOLID 原则", "观察者模式", "互斥锁", "epoll", "系统调用",
                         "Docker 镜像", "高阶函数", "负载均衡", "缓存一致性", "SQL 注入",
                         "页面置换算法", "拥塞控制", "跳表", "贪心算法", "MVCC",
                         "Paxos 协议", "语法分析", "梯度消失", "数字签名", "深度缓冲"],
            "other_concept": ["物理内存", "UDP", "AVL 树", "分治法", "哈希索引",
                               "BASE 理论", "语义分析", "前向传播", "AES", "光线追踪",
                               "单一职责", "策略模式", "自旋锁", "select", "用户态调用",
                               "容器卷", "闭包", "一致性哈希", "MESI 协议", "XSS"],
            "other_subject": ["操作系统", "计算机网络", "数据结构", "算法",
                               "数据库", "分布式系统", "编译原理", "机器学习"],
            "question": ["核心原理是什么", "怎么入门", "有什么好书推荐", "面试常考吗",
                          "实际项目里怎么用", "学习路线是什么", "有哪些常见误区",
                          "需要什么前置知识", "有哪些经典面试题"],
            "book": ["CSAPP", "DDIA", "算法导论", "深入理解计算机系统", "设计模式",
                      "SICP", "代码大全", "重构", "Effective C++", "Clean Code",
                      "数据库系统概念", "编译原理（龙书）", "深入 Linux 内核",
                      "TCP/IP 详解", "编程珠玑", "计算机程序的构造和解释"],
            "chapter": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
            "takeaway": ["终于搞懂了", "有点懵", "收获很大", "需要再看一遍",
                          "豁然开朗", "感觉还是没理解透", "做了不少笔记", "想动手实现一下"],
            "problem": ["看不懂", "记不住", "不会用", "容易混淆",
                         "不知道怎么下手", "概念太多串了", "看书会做题不会",
                         "缺少实际项目经验", "不知道学了有什么用"],
            "answer": [
                "核心是空间换时间", "关键在于理解状态转移", "通过分层解耦复杂性",
                "本质是资源的抽象和管理", "用索引加速查询", "通过一致性哈希解决数据分布",
                "通过锁和条件变量实现同步", "通过编译器前端+后端分离实现",
                "通过链式求导反向传播梯度", "通过矩阵变换把 3D 投影到 2D",
                "核心是封装变化、面向接口编程", "通过时间换空间提高可用性",
                "通过悲观锁/乐观锁控制并发", "通过事件驱动避免阻塞",
                "通过系统调用陷入内核态执行", "通过联合文件系统实现分层",
                "通过不可变数据和纯函数简化推理", "通过分发请求到多台机器提高吞吐",
                "通过缓存一致性协议保证多核可见性", "通过参数化查询防止注入",
            ],
            "example": ["比如缓存就是空间换时间", "比如背包问题就是经典 DP",
                         "比如 TCP/IP 分层", "比如虚拟内存就是资源的抽象",
                         "比如 B+ 树索引加速范围查询", "比如 Redis 用一致性哈希分片",
                         "比如生产者-消费者用条件变量", "比如 GCC 前端生成 AST",
                         "比如 ResNet 用残差连接解决梯度消失", "比如 OpenGL 用 MVP 矩阵投影",
                         "比如策略模式就是封装变化", "比如主从复制就是时间换空间",
                         "比如 select for update 是悲观锁", "比如 Nginx 用 epoll",
                         "比如 read() 触发软中断", "比如 Docker 用 OverlayFS",
                         "比如 map/reduce 是纯函数", "比如 LVS 把请求分发到多台 RS",
                         "比如 MESI 保证多核缓存一致", "比如 PreparedStatement 防止 SQL 注入"],
        }
    },
    "tech_news": {
        "name": "技术资讯",
        "user_templates": [
            "你听说了 {news} 吗？{reaction}。",
            "{company} 发布了 {product}，{comment}。",
            "最近 {trend} 很火，你怎么看？",
            "{language} 新版本发布了，{feature} 好用吗？",
            "{tool} 和 {other_tool} 选哪个？",
            "{company} 开源了 {open_source_project}，值得用吗？",
            "最近在关注 {trend}，{question}？",
            "{tool} 出了新版本，{new_feature_comment}。",
            "你看 {company} 的 {product} 能取代 {competitor} 吗？",
            "{trend} 这个方向，{concern}？",
        ],
        "assistant_templates": [
            "是的，{analysis}。{opinion}。",
            "{product} 的亮点是 {analysis}。{opinion}。",
            "{trend} 确实在加速，{analysis}。{opinion}。",
            "{feature} {analysis}。{opinion}。",
            "看场景：{comparison}。{opinion}。",
            "开源项目 {analysis}。{opinion}。",
        ],
        "vars": {
            "news": ["GPT-5 发布", "苹果出了 Vision Pro", "Rust 进 Linux 内核",
                      "GitHub 被 Copilot 改版", "Google 拆分 AI 部门",
                      "Meta 开源 Llama 4", "微软收购了游戏公司", "Nvidia 市值超过苹果",
                      "Docker 改了收费策略", "Stack Overflow 被 Procom 收购",
                      "Linux 内核 6.10 发布", "Python 3.13 移除 GIL", "Rust 基金会独立运营",
                      "Chrome 内置 AI 模型", "AWS 降价了"],
            "company": ["OpenAI", "Google", "Apple", "Microsoft", "Anthropic",
                         "Meta", "Nvidia", "AWS", "Cloudflare", "Vercel",
                         "Docker", "JetBrains", "Mozilla", "Linux 基金会"],
            "product": ["Gemini 2.0", "Claude 4", "GPT-5", "Copilot X",
                         "Llama 4", "Vision Pro 2", "Cursor Pro", "Docker Desktop 5",
                         "Claude Code", "GitHub Models", "AWS Bedrock", "Vercel AI SDK"],
            "trend": ["AI Agent", "RAG", "向量数据库", "大模型微调", "多模态",
                       "AI 编程助手", "边缘计算", "WebGPU", "WebAssembly",
                       "Serverless", "可观测性", "零信任安全", "AI 推理优化",
                       "小语言模型", "AI 搜索"],
            "language": ["Python 3.13", "C++23", "Rust 1.80", "Go 1.23",
                          "TypeScript 5.5", "Java 22", "Kotlin 2.0", "Swift 6"],
            "feature": ["协程", "模式匹配", "类型推断", "模块系统",
                         "GIL-free 模式", "生命周期改进", "泛型参数", "并发模型",
                         "宏改进", "内存安全", "管道操作符", "结构化并发"],
            "tool": ["Docker", "VS Code", "Vim", "JetBrains", "Cursor",
                      "Neovim", "Emacs", "Sublime", "Zed", "Cline",
                      "Copilot", "Claude Code"],
            "other_tool": ["Podman", "Cursor", "Neovim", "Eclipse", "Vim",
                            "VS Code", "Sublime", "JetBrains", "Copilot", "Zed",
                            "Copilot", "Cline"],
            "open_source_project": ["Llama 4", "Sveltia", "Bun 2.0", "Tauri 2",
                                     "Zed 编辑器", "Ollama", "LangChain 4", "vLLM",
                                     "Lottie", "Drizzle ORM"],
            "competitor": ["Copilot", "ChatGPT", "Docker", "VS Code", "Notion"],
            "reaction": ["感觉变化很大", "有点意外", "期待已久", "不太关注",
                          "终于来了", "感觉被背刺了", "意料之中", "想试试看"],
            "comment": ["看起来不错", "噱头大于实际", "值得关注", "价格太贵了",
                         "等评测再说", "赶紧升级", "先观望", "有点心动"],
            "new_feature_comment": ["新增的功能挺实用", "变化不大", "性能提升明显",
                                     "有点鸡肋", "终于支持了"],
            "concern": ["落地难度大不大", "就业前景怎么样", "现在入局来得及吗",
                        "和现有方案比有什么优势", "学习成本高不高"],
            "question": ["有什么实际应用场景", "和之前的方案比优势在哪", "现在学来得及吗",
                         "有哪些坑", "值不值得投入精力"],
            "analysis": [
                "主要提升了推理能力", "多模态融合是关键方向", "性能提升显著但生态还在跟进",
                "降低了使用门槛", "和竞品差距在缩小", "开源降低了试错成本",
                "推理成本降了一个数量级", "开发者体验大幅改善", "但生产环境还需要验证",
                "社区热度很高但成熟度不够", "底层优化为主，API 变化不大",
            ],
            "opinion": ["值得跟进", "观望就好", "可以试用", "等生态成熟再说",
                         "建议早学", "看团队需求", "短期内不会有影响", "长期看是趋势"],
            "comparison": ["Docker 生态更成熟，Podman 更轻量", "VS Code 插件多，Cursor AI 能力强",
                            "JetBrains 功能全，VS Code 更轻量", "Copilot 集成好，Claude Code 推理强",
                            "Vim 效率高但学习曲线陡，VS Code 上手快", "Neovim 生态比 Vim 更活跃",
                            "Zed 性能好但生态还不如 VS Code", "Cline 开源灵活，Copilot 省心"],
        }
    },
}


def _fill_template(template: str, vars_dict: dict) -> str:
    """用 vars_dict 中的值填充模板中的 {placeholder}"""
    result = template
    for key, values in vars_dict.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, random.choice(values))
    return result


def generate_conversations(count: int = 10000) -> list[dict]:
    """
    生成 count 条不重复的模拟对话。
    用 user 消息去重，如果组合数不够则返回实际唯一条数。
    """
    theme_keys = list(THEMES.keys())
    conversations = []
    seen_user_msgs = set()

    # 多生成 50% 的候选，去重后取前 count 条
    max_attempts = count * 3
    base_time = datetime.now() - timedelta(days=90)

    for i in range(max_attempts):
        if len(conversations) >= count:
            break

        theme_key = random.choice(theme_keys)
        theme = THEMES[theme_key]

        user_template = random.choice(theme["user_templates"])
        assistant_template = random.choice(theme["assistant_templates"])

        user_msg = _fill_template(user_template, theme["vars"])
        assistant_msg = _fill_template(assistant_template, theme["vars"])

        # 去重：user 消息不能重复
        if user_msg in seen_user_msgs:
            continue
        seen_user_msgs.add(user_msg)

        timestamp = (base_time + timedelta(seconds=i * 100)).isoformat()

        conversations.append({
            "user": user_msg,
            "assistant": assistant_msg,
            "topic": theme_key,
            "topic_name": theme["name"],
            "timestamp": timestamp,
            "index": len(conversations),
        })

    return conversations


def save_conversations(conversations: list[dict], path: str):
    """保存到 JSON"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)
    print(f"已生成 {len(conversations)} 条对话 -> {path}")

    # 去重检查
    user_msgs = [c["user"] for c in conversations]
    unique = len(set(user_msgs))
    print(f"唯一 user 消息: {unique}/{len(conversations)}")

    # 统计主题分布
    topic_counts = {}
    for c in conversations:
        name = c["topic_name"]
        topic_counts[name] = topic_counts.get(name, 0) + 1
    print("主题分布:")
    for name, cnt in sorted(topic_counts.items()):
        print(f"  {name}: {cnt} 条")


if __name__ == "__main__":
    import config
    conversations = generate_conversations(config.DATA_SIZE)
    save_path = os.path.join(os.path.dirname(__file__), "data", "conversations.json")
    save_conversations(conversations, save_path)
