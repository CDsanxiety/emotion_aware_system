class PADEmotionEngine:
    def __init__(self):
        self.P = 0.0 # 中性
        self.A = 0.0
        self.D = 0.0
        
    def update(self, user_sentiment_score, voice_volume_score, emotion_type=None):
        """
        每次交互更新机器人的内心情感状态
        例如：用户大声抱怨 (volume高，sentiment负)，会降低机器人的 P，拉高 A
        """
        decay_rate = 0.1 # 情绪随时间向中性衰减
        
        # 简单的情感计算公式 
        self.P = (1 - decay_rate) * self.P + 0.3 * user_sentiment_score
        self.A = (1 - decay_rate) * self.A + 0.2 * voice_volume_score
        
        # 根据情绪类型更新D值
        if emotion_type:
            # 不同情绪类型对支配度的影响
            emotion_to_dominance = {
                "happy": 0.2,    # 开心时支配度略有提升
                "sad": -0.3,     # 悲伤时支配度降低
                "angry": 0.5,    # 愤怒时支配度明显提升
                "fear": -0.5,    # 恐惧时支配度明显降低
                "neutral": 0.0,  # 中性时支配度不变
                "surprise": 0.1, # 惊讶时支配度略有提升
                "disgust": -0.2  # 厌恶时支配度降低
            }
            d_change = emotion_to_dominance.get(emotion_type, 0.0)
            self.D = (1 - decay_rate) * self.D + 0.2 * d_change
        else:
            # 如果没有情绪类型，D值自然衰减
            self.D = (1 - decay_rate) * self.D
        
        # 限制在 [-1, 1] 区间
        self.P = max(-1.0, min(1.0, self.P))
        self.A = max(-1.0, min(1.0, self.A))
        self.D = max(-1.0, min(1.0, self.D))
        
    def get_tts_params(self):
        """将 PAD 映射到 TTS 引擎的参数"""
        speed = "+0%"
        pitch = "+0Hz"
        
        if self.A > 0.5:
            speed = "+15%" # 唤醒度高，语速变快
        elif self.A < -0.5:
            speed = "-15%" # 唤醒度低（疲惫），语速变慢
            
        if self.P > 0.5:
            pitch = "+10Hz" # 开心时音调变高
            
        return {"rate": speed, "pitch": pitch}