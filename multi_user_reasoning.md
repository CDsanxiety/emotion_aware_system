# 多用户推理逻辑展示

## 用户: user_001

# Reasoning Graph: b3391ba1

**创建时间**: 2026-04-19 15:28:27
**用户ID**: user_001
**节点数**: 10
**边数**: 8

## 节点

| 节点ID | 类型 | 标签 | 数据 |
|--------|------|------|------|
| raw_sensor_0 | raw_sensor | 原始camera | {"sensor_type": "camera", "raw_data": "face_detect |
| perception_1 | perception | 感知细节 | {"description": "检测到主人的面部", "confidence": 0.9} |
| scene_understanding_2 | scene_understanding | 场景: 家庭场景 | {"scene_type": "家庭场景", "confidence": 0.8, "keyword |
| memory_association_3 | memory_association | 记忆关联: identity | {"memory_type": "identity", "content": "识别为陌生人", " |
| memory_association_4 | memory_association | 记忆关联: health | {"memory_type": "health", "content": "主人感冒还没好", "r |
| emotion_detection_5 | emotion_detection | 情绪: neutral | {"emotion": "neutral", "confidence": 0.4, "pad_sta |
| uncertainty_reasoning_6 | uncertainty_reasoning | 决策模式: query | {"decision_mode": "query", "confidence": 0.4, "rea |
| intent_decision_7 | intent_decision | 用户意图: 问候 | {"intent": "问候", "confidence": 0.95, "alternatives |
| action_selection_8 | action_selection | 选择动作: 问候主人并询问感冒情况 | {"selected_action": "问候主人并询问感冒情况", "llm_suggestion |
| final_action_9 | final_action | 执行: 问候主人并询问感冒情况 | {"action": "问候主人并询问感冒情况", "success": true, "execut |

## 边 (逻辑链路)

| 源节点 | 目标节点 | 类型 | 标签 |
|--------|---------|------|------|
| raw_sensor_0 | perception_1 | causes | 感知来源 |
| perception_1 | scene_understanding_2 | supports | 场景推断 |
| perception_1 | emotion_detection_5 | causes | 情绪推断 |
| emotion_detection_5 | uncertainty_reasoning_6 | causes | 不确定性评估 |
| intent_decision_7 | action_selection_8 | leads_to | 意图驱动 |
| action_selection_8 | final_action_9 | leads_to | 执行动作 |
| scene_understanding_2 | final_action_9 | supports | 场景相关 |
| memory_association_4 | final_action_9 | supports | 记忆驱动 |

## 决策结论

- **执行: 问候主人并询问感冒情况**: {"action": "问候主人并询问感冒情况", "success": true, "execution_details": {"response": "完成"}}

---

## 用户: stranger_001

# Reasoning Graph: 986f88e2

**创建时间**: 2026-04-19 15:28:28
**用户ID**: stranger_001
**节点数**: 9
**边数**: 8

## 节点

| 节点ID | 类型 | 标签 | 数据 |
|--------|------|------|------|
| raw_sensor_0 | raw_sensor | 原始camera | {"sensor_type": "camera", "raw_data": "face_detect |
| perception_1 | perception | 感知细节 | {"description": "检测到陌生人的面部", "confidence": 0.9} |
| scene_understanding_2 | scene_understanding | 场景: 陌生场景 | {"scene_type": "陌生场景", "confidence": 0.8, "keyword |
| memory_association_3 | memory_association | 记忆关联: identity | {"memory_type": "identity", "content": "识别为陌生人", " |
| emotion_detection_4 | emotion_detection | 情绪: neutral | {"emotion": "neutral", "confidence": 0.5, "pad_sta |
| uncertainty_reasoning_5 | uncertainty_reasoning | 决策模式: uncertain | {"decision_mode": "uncertain", "confidence": 0.5,  |
| intent_decision_6 | intent_decision | 用户意图: 问候 | {"intent": "问候", "confidence": 0.95, "alternatives |
| action_selection_7 | action_selection | 选择动作: 礼貌问候陌生人 | {"selected_action": "礼貌问候陌生人", "llm_suggestion": " |
| final_action_8 | final_action | 执行: 礼貌问候陌生人 | {"action": "礼貌问候陌生人", "success": true, "execution_ |

## 边 (逻辑链路)

| 源节点 | 目标节点 | 类型 | 标签 |
|--------|---------|------|------|
| raw_sensor_0 | perception_1 | causes | 感知来源 |
| perception_1 | scene_understanding_2 | supports | 场景推断 |
| perception_1 | emotion_detection_4 | causes | 情绪推断 |
| emotion_detection_4 | uncertainty_reasoning_5 | causes | 不确定性评估 |
| intent_decision_6 | action_selection_7 | leads_to | 意图驱动 |
| action_selection_7 | final_action_8 | leads_to | 执行动作 |
| scene_understanding_2 | final_action_8 | supports | 场景相关 |
| memory_association_3 | final_action_8 | supports | 记忆驱动 |

## 决策结论

- **执行: 礼貌问候陌生人**: {"action": "礼貌问候陌生人", "success": true, "execution_details": {"response": "完成"}}

---
