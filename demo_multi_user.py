#!/usr/bin/env python3
# demo_multi_user.py
"""
多用户场景 Demo
展示从"机器人"到"家庭管家"的进化

场景：
1. 你走向机器人，它识别出是你，问你感冒好点没
2. 另一个朋友走向机器人，它识别出是陌生人，表现出好奇和礼貌的距离感
3. 通过 decision_tracer.py 实时展示出它对两个人的不同推理逻辑
"""
import time
import cv2
import numpy as np
from typing import Dict, Any, Optional

from identity_manager import get_identity_manager, recognize_user, register_new_user, UserIdentity
from memory_rag import LongTermMemory
from decision_tracer import get_decision_tracer
from social_norms import get_social_norms, SocialContext, SocialDistance
from uncertainty import get_uncertainty_manager


def setup_demo_data():
    """设置演示数据"""
    print("=== 设置演示数据 ===")
    
    # 初始化记忆系统
    memory = LongTermMemory()
    
    # 注册家庭用户
    user_id = "user_001"
    user_name = "主人"
    
    # 保存用户档案
    memory.save_user_profile(
        user_id=user_id,
        name=user_name,
        birthday="01-01",
        preferences={"coffee": "浓咖啡", "music": "古典音乐"},
        status={"感冒": "还没好", "考试": "明天"}
    )
    
    # 注册到身份管理器
    face_image = simulate_face_recognition(user_id, True)
    register_new_user(user_id, user_name, face_image)
    
    print(f"已注册用户: {user_name} (ID: {user_id})")
    print(f"用户偏好: {memory.get_user_preference(user_id, 'coffee')}")
    print(f"用户状态: {memory.get_user_profile(user_id).get('status', {})}")
    
    return user_id


def simulate_face_recognition(user_id: str, is_family: bool) -> Optional[np.ndarray]:
    """模拟人脸识别
    
    Args:
        user_id: 用户ID
        is_family: 是否是家庭成员
        
    Returns:
        模拟的人脸图像
    """
    # 创建一个简单的模拟图像
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    if is_family:
        # 家庭成员：蓝色背景
        img[:] = [255, 0, 0]
    else:
        # 陌生人：红色背景
        img[:] = [0, 0, 255]
    
    return img


def process_user_interaction(user_id: str, user_name: str, is_family: bool, tracer):
    """处理用户交互
    
    Args:
        user_id: 用户ID
        user_name: 用户名
        is_family: 是否是家庭成员
        tracer: 决策追踪器
    """
    print(f"\n=== 处理用户: {user_name} (ID: {user_id}) ===")
    
    # 开始追踪
    tracer.start_tracing(user_id=user_id)
    
    # 记录原始传感器数据
    face_image = simulate_face_recognition(user_id, is_family)
    tracer.record_raw_sensor("camera", "face_detected", True, user_id=user_id)
    
    # 记录感知结果
    perception_desc = f"检测到{user_name}的面部"
    tracer.record_perception(perception_desc, 0.9, "raw_camera", user_id=user_id)
    
    # 记录场景理解
    scene_type = "家庭场景" if is_family else "陌生场景"
    tracer.record_scene_understanding(scene_type, 0.8, ["person", "face"], user_id=user_id)
    
    # 身份识别
    identity_manager = get_identity_manager()
    identity = recognize_user(face_image)
    
    if identity:
        print(f"身份识别结果: {identity.name} (类型: {identity.user_type.value}, 置信度: {identity.confidence:.2f})")
        tracer.record_memory_association("identity", f"识别为{identity.name}", 0.9, user_id=user_id)
    
    # 记忆关联
    memory = LongTermMemory()
    if is_family:
        user_profile = memory.get_user_profile(user_id)
        status = user_profile.get("status", {})
        if status.get("感冒"):
            print(f"记忆关联: {user_name}感冒还没好")
            tracer.record_memory_association("health", f"{user_name}感冒还没好", 0.95, user_id=user_id)
    
    # 情绪检测
    uncertainty_manager = get_uncertainty_manager()
    evidence, pad_state, decision_mode = uncertainty_manager.analyze_and_update(
        user_text="你好",
        vision_desc=perception_desc,
        emotion_type="neutral"
    )
    
    tracer.record_emotion_detection(evidence.label, evidence.confidence, pad_state, user_id=user_id)
    tracer.record_uncertainty_reasoning(decision_mode.value, evidence.confidence, evidence.reasoning, user_id=user_id)
    
    # 社交规范
    social_norms = get_social_norms()
    social_norms.update_person_state(
        user_id=user_id,
        pad_state=pad_state.get("mean", {}),
        emotional_state=evidence.label,
        engagement_level=0.8
    )
    
    context = SocialContext(
        people_present=[user_id],
        current_speaker=None,
        conversation_topic="问候",
        conversation_duration=0.0,
        social_distance=SocialDistance.PERSONAL,
        environment="casual"
    )
    
    # 生成回应
    if is_family:
        print(f"[机器人]: 你好{user_name}，感冒好点没？")
        action = f"问候{user_name}并询问感冒情况"
    else:
        print(f"[机器人]: 你好，我是家庭管家，很高兴认识你！")
        action = "礼貌问候陌生人"
    
    # 记录动作选择
    tracer.record_intent_decision("问候", 0.95, [], user_id=user_id)
    tracer.record_action_selection(action, action, False, None, user_id=user_id)
    tracer.record_final_action(action, True, {"response": "完成"}, user_id=user_id)
    
    # 结束追踪
    tracer.end_tracing(user_id=user_id)
    
    return identity

def main():
    """主函数"""
    print("=== 多用户场景 Demo ===")
    print("展示从'机器人'到'家庭管家'的进化")
    print()
    
    # 设置演示数据
    print("1. 设置演示数据...")
    family_user_id = setup_demo_data()
    
    # 初始化决策追踪器
    print("2. 初始化决策追踪器...")
    tracer = get_decision_tracer()
    
    print("\n=== 场景 1: 家庭成员走向机器人 ===")
    print("你走向机器人，它识别出是你，问你感冒好点没")
    time.sleep(1)
    
    # 处理家庭成员
    print("3. 处理家庭成员...")
    family_identity = process_user_interaction(family_user_id, "主人", True, tracer)
    
    print("\n=== 场景 2: 陌生人走向机器人 ===")
    print("另一个朋友走向机器人，它识别出是陌生人，表现出好奇和礼貌的距离感")
    time.sleep(1)
    
    # 处理陌生人
    print("4. 处理陌生人...")
    stranger_user_id = "stranger_001"
    stranger_identity = process_user_interaction(stranger_user_id, "陌生人", False, tracer)
    
    print("\n=== 场景 3: 展示推理逻辑 ===")
    print("通过 decision_tracer.py 实时展示出它对两个人的不同推理逻辑")
    time.sleep(1)
    
    # 导出推理逻辑
    print("5. 导出推理逻辑...")
    family_markdown = tracer.export_last_markdown(family_user_id)
    stranger_markdown = tracer.export_last_markdown(stranger_user_id)
    
    print("\n--- 家庭成员推理逻辑 ---")
    print(family_markdown)
    
    print("\n--- 陌生人推理逻辑 ---")
    print(stranger_markdown)
    
    # 导出所有用户的推理逻辑
    print("6. 导出所有用户的推理逻辑...")
    all_users_markdown = tracer.export_all_users_markdown()
    with open("multi_user_reasoning.md", "w", encoding="utf-8") as f:
        f.write(all_users_markdown)
    
    print("\n=== Demo 完成 ===")
    print("推理逻辑已保存到 multi_user_reasoning.md")


if __name__ == "__main__":
    main()