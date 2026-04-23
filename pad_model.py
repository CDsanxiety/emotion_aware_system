<<<<<<< HEAD
class PADEmotionEngine:
    def __init__(self):
        self.P = 0.0 # 中性
        self.A = 0.0
        self.D = 0.0
        self._expression_controller = None

    def bind_expression_controller(self, controller):
        """绑定物理表达控制器"""
        self._expression_controller = controller

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
                "disgust": -0.2,  # 厌恶时支配度降低
                "caring": 0.0,   # 关怀时支配度中性
                "supportive": -0.1, # 支持时略微顺从
                "empathetic": -0.2, # 共情时顺从
                "helpful": 0.1   # 帮助时略微自信
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

        # 更新物理表达
        self._notify_physical_expression()

    def _notify_physical_expression(self):
        """通知物理表达系统更新"""
        if self._expression_controller:
            self._expression_controller.update_emotion(self.P, self.A, self.D)

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

    def get_physical_expression(self):
        """获取当前 PAD 对应的物理表达参数"""
        from physical_expression import create_expression_from_pad
        return create_expression_from_pad(self.P, self.A, self.D)

    def get_pad_values(self):
        """获取当前 PAD 值"""
        return {"P": self.P, "A": self.A, "D": self.D}

    def get_emotion_state_name(self):
        """获取情绪状态名称"""
        if self.P > 0.5:
            if self.A > 0.5:
                return "开心兴奋"
            else:
                return "放松愉悦"
        elif self.P < -0.5:
            if self.A > 0.5:
                return "紧张不安"
            else:
                return "悲伤疲惫"
        else:
            if self.A > 0.5:
                return "警觉好奇"
            else:
=======
class PADEmotionEngine:
    def __init__(self):
        self.P = 0.0 # 中性
        self.A = 0.0
        self.D = 0.0
        self._expression_controller = None

    def bind_expression_controller(self, controller):
        """绑定物理表达控制器"""
        self._expression_controller = controller

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
                "disgust": -0.2,  # 厌恶时支配度降低
                "caring": 0.0,   # 关怀时支配度中性
                "supportive": -0.1, # 支持时略微顺从
                "empathetic": -0.2, # 共情时顺从
                "helpful": 0.1   # 帮助时略微自信
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

        # 更新物理表达
        self._notify_physical_expression()

    def _notify_physical_expression(self):
        """通知物理表达系统更新"""
        if self._expression_controller:
            self._expression_controller.update_emotion(self.P, self.A, self.D)

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

    def get_physical_expression(self):
        """获取当前 PAD 对应的物理表达参数"""
        from physical_expression import create_expression_from_pad
        return create_expression_from_pad(self.P, self.A, self.D)

    def get_pad_values(self):
        """获取当前 PAD 值"""
        return {"P": self.P, "A": self.A, "D": self.D}

    def get_emotion_state_name(self):
        """获取情绪状态名称"""
        if self.P > 0.5:
            if self.A > 0.5:
                return "开心兴奋"
            else:
                return "放松愉悦"
        elif self.P < -0.5:
            if self.A > 0.5:
                return "紧张不安"
            else:
                return "悲伤疲惫"
        else:
            if self.A > 0.5:
                return "警觉好奇"
            else:
>>>>>>> 483f2a96306b03f52efde3fc5895cf74d9121b3f
                return "平静中性"