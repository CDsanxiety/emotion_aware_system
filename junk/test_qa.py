# test_qa.py - 暴力 QA 测试脚本
from llm_api import get_response
import json
from datetime import datetime

# 测试用例库（表情 + 用户说的话）
test_cases = [
    # 开心场景
    ("happy", "我中奖了！太开心了！"),
    ("happy", "今天天气真好，心情也好"),
    ("happy", "刚收到一份超棒的礼物"),
    
    # 难过场景
    ("sad", "我失恋了，好难过..."),
    ("sad", "刚和好朋友吵架了"),
    ("sad", "今天工作好累，心累"),
    
    # 愤怒场景
    ("angry", "今天被老板骂了，气死我了！"),
    ("angry", "有人插队，素质真差！"),
    ("angry", "外卖送了一个小时还没到"),
    
    # 惊讶场景
    ("surprise", "哇！你居然会说话？"),
    ("surprise", "天呐，外面下雪了！"),
    ("surprise", "你居然知道我心情不好？"),
    
    # 害怕场景
    ("fear", "刚才路上看到一条蛇，吓死我了"),
    ("fear", "做了一个噩梦..."),
    ("fear", "外面打雷好大声"),
    
    # 平静场景
    ("neutral", "今天午饭吃什么好呢"),
    ("neutral", "帮我想想周末去哪玩"),
    ("neutral", ""),  # 不说话的情况
    
    # 厌恶场景
    ("disgust", "这菜太难吃了"),
    ("disgust", "地铁上好臭啊"),
]

def run_qa_test():
    print("=" * 60)
    print(f"🧪 暴力 QA 测试开始 - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    
    issues = []  # 记录有问题的回复
    
    for i, (emotion, text) in enumerate(test_cases, 1):
        print(f"\n[测试 {i}/{len(test_cases)}]")
        print(f"  输入表情: {emotion}")
        print(f"  用户说: '{text}'" if text else "  用户说: (沉默)")
        
        try:
            result = get_response(emotion, text, enable_tts=False)
            
            print(f"  📊 分析情绪: {result.get('emotion')}")
            print(f"  🎬 家居动作: {result.get('action')}")
            print(f"  💬 暖暖回复: {result.get('reply')}")
            
            # 检查潜在问题
            reply = result.get('reply', '')
            action = result.get('action', '')
            
            if len(reply) > 50:
                issues.append(f"[{i}] 回复太长 ({len(reply)}字): {reply}")
                print(f"  ⚠️ 警告: 回复超过50字")
            
            if action not in ['播放音乐', '调节灯光', '无动作']:
                issues.append(f"[{i}] 无效动作: {action}")
                print(f"  ⚠️ 警告: 动作不在预设选项中")
            
            if not reply and text:
                issues.append(f"[{i}] 有输入但回复为空")
                print(f"  ⚠️ 警告: 回复为空")
                
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            issues.append(f"[{i}] 异常: {e}")
        
        print("-" * 40)
    
    # 总结报告
    print("\n" + "=" * 60)
    print("📋 测试总结")
    print("=" * 60)
    print(f"  总测试数: {len(test_cases)}")
    print(f"  发现问题: {len(issues)}")
    
    if issues:
        print("\n⚠️ 发现的问题:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("  ✅ 所有测试通过！回复质量良好！")
    
    print("=" * 60)
    return issues

if __name__ == "__main__":
    run_qa_test()