# identity_manager.py
"""
身份识别模块
支持人脸识别和声纹识别，为每个家庭成员建立独立的 User Profile
"""
import os
import time
import uuid
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

from utils import logger

# 尝试导入声纹识别相关库
try:
    import librosa
    import soundfile as sf
    _librosa_available = True
except ImportError:
    _librosa_available = False
    logger.warning("librosa not available, voice recognition disabled")


class RecognitionMethod(Enum):
    FACE = "face"
    VOICE = "voice"
    BOTH = "both"


class UserType(Enum):
    FAMILY = "family"
    FRIEND = "friend"
    STRANGER = "stranger"


class RecognitionEngineType(Enum):
    TRADITIONAL = "traditional"  # LBPH 等传统算法
    DEEP_LEARNING = "deep_learning"  # InsightFace、MobileFaceNet 等深度学习算法


@dataclass
class UserIdentity:
    """用户身份信息"""
    user_id: str
    name: str
    user_type: UserType
    confidence: float
    recognition_method: RecognitionMethod
    metadata: Dict[str, Any] = None


class IdentityManager:
    """
    身份识别管理器
    支持人脸识别和声纹识别
    """
    def __init__(self, data_dir: str = "./identity_data",
                 face_engine_type: RecognitionEngineType = RecognitionEngineType.TRADITIONAL):
        self.data_dir = data_dir
        self.face_data_dir = os.path.join(data_dir, "face")
        self.voice_data_dir = os.path.join(data_dir, "voice")
        self.face_engine_type = face_engine_type
        self._ensure_directories()
        
        # 加载人脸识别模型
        self.face_detector = self._load_face_detector()
        self.face_recognizer = self._load_face_recognizer()
        self.deep_face_recognizer = self._load_deep_face_recognizer()
        
        # 声纹模型参数
        self.voice_features = {}
        self._load_voice_data()
        
        # 人脸特征存储（用于深度学习）
        self.face_features = {}
        self._load_face_features()
    
    def _ensure_directories(self):
        """确保数据目录存在"""
        os.makedirs(self.face_data_dir, exist_ok=True)
        os.makedirs(self.voice_data_dir, exist_ok=True)
    
    def _load_face_detector(self):
        """加载人脸检测器"""
        try:
            # 使用 OpenCV 的 Haar 级联分类器
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                return cv2.CascadeClassifier(cascade_path)
            logger.warning("Haar cascade not found, using dnn detector")
            # 回退到 DNN 检测器
            model_path = "res10_300x300_ssd_iter_140000.caffemodel"
            config_path = "deploy.prototxt.txt"
            if os.path.exists(model_path) and os.path.exists(config_path):
                return cv2.dnn.readNetFromCaffe(config_path, model_path)
            return None
        except Exception as e:
            logger.error(f"Failed to load face detector: {e}")
            return None
    
    def _load_face_recognizer(self):
        """加载人脸识别器"""
        try:
            # 使用 OpenCV 的 LBPH 人脸识别器
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            # 尝试加载训练数据
            train_data_path = os.path.join(self.face_data_dir, "trainer.yml")
            if os.path.exists(train_data_path):
                recognizer.read(train_data_path)
            return recognizer
        except Exception as e:
            logger.error(f"Failed to load face recognizer: {e}")
            return None
    
    def _load_deep_face_recognizer(self):
        """加载深度学习人脸识别器（如 InsightFace 或 MobileFaceNet）"""
        try:
            # 尝试导入 InsightFace
            try:
                import insightface
                from insightface.app import FaceAnalysis
                app = FaceAnalysis(name='buffalo_l')
                app.prepare(ctx_id=0, det_size=(640, 640))
                logger.info("InsightFace loaded successfully")
                return app
            except ImportError:
                logger.warning("InsightFace not available, falling back to MobileFaceNet")
                
            # 尝试导入 MobileFaceNet
            try:
                import torch
                from facenet_pytorch import MTCNN, InceptionResnetV1
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
                logger.info("MobileFaceNet loaded successfully")
                return resnet
            except ImportError:
                logger.warning("Deep learning face recognizers not available")
                return None
        except Exception as e:
            logger.error(f"Failed to load deep face recognizer: {e}")
            return None
    
    def _load_face_features(self):
        """加载人脸特征数据（用于深度学习）"""
        try:
            for filename in os.listdir(self.face_data_dir):
                if filename.endswith(".npy"):
                    user_id = filename[:-4]
                    feature_path = os.path.join(self.face_data_dir, filename)
                    self.face_features[user_id] = np.load(feature_path)
        except Exception as e:
            logger.error(f"Failed to load face features: {e}")
    
    def _load_voice_data(self):
        """加载声纹数据"""
        try:
            for filename in os.listdir(self.voice_data_dir):
                if filename.endswith(".npy"):
                    user_id = filename[:-4]
                    feature_path = os.path.join(self.voice_data_dir, filename)
                    self.voice_features[user_id] = np.load(feature_path)
        except Exception as e:
            logger.error(f"Failed to load voice data: {e}")
    
    def register_user(self, user_id: str, name: str, user_type: UserType,
                      face_image: Optional[np.ndarray] = None,
                      voice_audio: Optional[np.ndarray] = None,
                      sample_rate: int = 22050) -> bool:
        """注册新用户
        
        Args:
            user_id: 用户唯一标识
            name: 用户名
            user_type: 用户类型
            face_image: 人脸图像
            voice_audio: 声纹音频
            sample_rate: 音频采样率
            
        Returns:
            是否注册成功
        """
        try:
            # 注册人脸
            if face_image is not None:
                self._register_face(user_id, face_image)
            
            # 注册声纹
            if voice_audio is not None:
                self._register_voice(user_id, voice_audio, sample_rate)
            
            return True
        except Exception as e:
            logger.error(f"Failed to register user: {e}")
            return False
    
    def _register_face(self, user_id: str, face_image: np.ndarray):
        """注册人脸"""
        # 检测人脸
        faces = self._detect_faces(face_image)
        if not faces:
            raise ValueError("No face detected")
        
        # 根据引擎类型选择注册方法
        if self.face_engine_type == RecognitionEngineType.DEEP_LEARNING and self.deep_face_recognizer:
            self._register_face_deep(user_id, face_image, faces[0])
        else:
            self._register_face_traditional(user_id, face_image, faces[0])
    
    def _register_face_traditional(self, user_id: str, face_image: np.ndarray, face_bbox: Tuple[int, int, int, int]):
        """使用传统方法注册人脸（LBPH）"""
        # 提取人脸区域
        x, y, w, h = face_bbox
        face_roi = face_image[y:y+h, x:x+w]
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi = cv2.resize(face_roi, (100, 100))
        
        # 保存人脸数据
        face_path = os.path.join(self.face_data_dir, f"{user_id}.jpg")
        cv2.imwrite(face_path, face_roi)
        
        # 更新人脸识别模型
        self._train_face_recognizer()
    
    def _register_face_deep(self, user_id: str, face_image: np.ndarray, face_bbox: Tuple[int, int, int, int]):
        """使用深度学习方法注册人脸"""
        # 提取人脸特征
        face_feature = self._extract_face_feature(face_image, face_bbox)
        if face_feature is None:
            raise ValueError("Failed to extract face feature")
        
        # 保存人脸特征
        feature_path = os.path.join(self.face_data_dir, f"{user_id}.npy")
        np.save(feature_path, face_feature)
        
        # 更新内存中的特征
        self.face_features[user_id] = face_feature
        
        # 同时保存人脸图像作为备份
        x, y, w, h = face_bbox
        face_roi = face_image[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (100, 100))
        face_path = os.path.join(self.face_data_dir, f"{user_id}.jpg")
        cv2.imwrite(face_path, face_roi)
    
    def _register_voice(self, user_id: str, audio: np.ndarray, sample_rate: int):
        """注册声纹"""
        if not _librosa_available:
            logger.warning("librosa not available, voice registration skipped")
            return
        
        # 提取声纹特征
        feature = self._extract_voice_features(audio, sample_rate)
        
        # 保存声纹特征
        feature_path = os.path.join(self.voice_data_dir, f"{user_id}.npy")
        np.save(feature_path, feature)
        
        # 更新内存中的特征
        self.voice_features[user_id] = feature
    
    def _train_face_recognizer(self):
        """训练人脸识别器"""
        if not self.face_recognizer:
            return
        
        faces = []
        labels = []
        label_map = {}
        current_label = 0
        
        for filename in os.listdir(self.face_data_dir):
            if filename.endswith(".jpg"):
                user_id = filename[:-4]
                if user_id not in label_map:
                    label_map[user_id] = current_label
                    current_label += 1
                
                image_path = os.path.join(self.face_data_dir, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    faces.append(image)
                    labels.append(label_map[user_id])
        
        if faces and labels:
            self.face_recognizer.train(faces, np.array(labels))
            trainer_path = os.path.join(self.face_data_dir, "trainer.yml")
            self.face_recognizer.save(trainer_path)
    
    def recognize_identity(self, face_image: Optional[np.ndarray] = None,
                         voice_audio: Optional[np.ndarray] = None,
                         sample_rate: int = 22050) -> Optional[UserIdentity]:
        """识别用户身份
        
        Args:
            face_image: 人脸图像
            voice_audio: 声纹音频
            sample_rate: 音频采样率
            
        Returns:
            用户身份信息
        """
        face_result = None
        voice_result = None
        
        # 人脸识别
        if face_image is not None and self.face_detector:
            face_result = self._recognize_face(face_image)
        
        # 声纹识别
        if voice_audio is not None and self.voice_features:
            voice_result = self._recognize_voice(voice_audio, sample_rate)
        
        # 融合结果
        return self._fuse_results(face_result, voice_result)
    
    def _detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """检测人脸"""
        if not self.face_detector:
            return []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 检查是否是 Haar 分类器
            if hasattr(self.face_detector, 'detectMultiScale'):
                faces = self.face_detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                return faces.tolist() if len(faces) > 0 else []
            
            # DNN 检测器
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
            )
            self.face_detector.setInput(blob)
            detections = self.face_detector.forward()
            
            faces = []
            h, w = image.shape[:2]
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x1, y1) = box.astype(int)
                    faces.append((x, y, x1-x, y1-y))
            return faces
        except Exception as e:
            logger.error(f"Failed to detect faces: {e}")
            return []
    
    def _recognize_face(self, image: np.ndarray) -> Optional[Tuple[str, float]]:
        """识别人脸"""
        faces = self._detect_faces(image)
        if not faces:
            return None
        
        # 根据引擎类型选择识别方法
        if self.face_engine_type == RecognitionEngineType.DEEP_LEARNING and self.deep_face_recognizer:
            return self._recognize_face_deep(image, faces)
        else:
            return self._recognize_face_traditional(image, faces)
    
    def _recognize_face_traditional(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> Optional[Tuple[str, float]]:
        """使用传统方法识别人脸（LBPH）"""
        if not self.face_recognizer:
            return None
        
        x, y, w, h = faces[0]
        face_roi = image[y:y+h, x:x+w]
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi = cv2.resize(face_roi, (100, 100))
        
        try:
            label, confidence = self.face_recognizer.predict(face_roi)
            # 查找用户ID
            user_id = self._get_user_id_from_label(label)
            if user_id and confidence < 100:
                return user_id, 1.0 - (confidence / 100.0)
        except Exception as e:
            logger.error(f"Failed to recognize face with traditional method: {e}")
        
        return None
    
    def _recognize_face_deep(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> Optional[Tuple[str, float]]:
        """使用深度学习方法识别人脸（InsightFace 或 MobileFaceNet）"""
        if not self.deep_face_recognizer or not self.face_features:
            return None
        
        try:
            # 提取人脸特征
            face_feature = self._extract_face_feature(image, faces[0])
            if face_feature is None:
                return None
            
            # 计算与已知特征的相似度
            best_user = None
            best_score = -float('inf')
            
            for user_id, ref_feature in self.face_features.items():
                # 计算余弦相似度
                similarity = self._calculate_similarity(face_feature, ref_feature)
                if similarity > best_score:
                    best_score = similarity
                    best_user = user_id
            
            # 阈值判断
            if best_user and best_score > 0.5:
                return best_user, best_score
        except Exception as e:
            logger.error(f"Failed to recognize face with deep learning: {e}")
        
        return None
    
    def _calculate_similarity(self, feature1: np.ndarray, feature2: np.ndarray) -> float:
        """计算两个特征向量的相似度"""
        try:
            # 归一化特征向量
            feature1 = feature1 / np.linalg.norm(feature1)
            feature2 = feature2 / np.linalg.norm(feature2)
            # 计算余弦相似度
            similarity = np.dot(feature1, feature2)
            return similarity
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    def _extract_face_feature(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """提取人脸特征"""
        try:
            x, y, w, h = face_bbox
            face_roi = image[y:y+h, x:x+w]
            
            # 根据使用的深度学习模型类型提取特征
            if hasattr(self.deep_face_recognizer, 'get'):  # InsightFace
                # InsightFace 直接处理整个图像
                faces = self.deep_face_recognizer.get(image)
                if faces and len(faces) > 0:
                    return faces[0].normed_embedding
            elif hasattr(self.deep_face_recognizer, 'forward'):  # MobileFaceNet (facenet-pytorch)
                import torch
                from torchvision import transforms
                
                # 预处理图像
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((160, 160)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                
                face_tensor = transform(face_roi).unsqueeze(0).to(next(self.deep_face_recognizer.parameters()).device)
                with torch.no_grad():
                    feature = self.deep_face_recognizer(face_tensor).cpu().numpy().flatten()
                return feature
        except Exception as e:
            logger.error(f"Failed to extract face feature: {e}")
        
        return None

    def _get_user_id_from_label(self, label: int) -> Optional[str]:
        """从标签获取用户ID"""
        label_map = {}
        current_label = 0
        
        for filename in os.listdir(self.face_data_dir):
            if filename.endswith(".jpg"):
                user_id = filename[:-4]
                if user_id not in label_map:
                    label_map[user_id] = current_label
                    current_label += 1
        
        # 反向查找
        for user_id, lbl in label_map.items():
            if lbl == label:
                return user_id
        return None
    
    def _extract_voice_features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """提取声纹特征"""
        if not _librosa_available:
            return np.array([])
        
        # 提取 MFCC 特征
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        # 计算均值和标准差
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        # 合并特征
        feature = np.hstack([mfcc_mean, mfcc_std])
        return feature
    
    def _recognize_voice(self, audio: np.ndarray, sample_rate: int) -> Optional[Tuple[str, float]]:
        """识别声纹"""
        if not self.voice_features:
            return None
        
        try:
            # 提取特征
            feature = self._extract_voice_features(audio, sample_rate)
            
            # 计算相似度
            best_user = None
            best_score = float('inf')
            
            for user_id, ref_feature in self.voice_features.items():
                # 计算欧氏距离
                distance = np.linalg.norm(feature - ref_feature)
                if distance < best_score:
                    best_score = distance
                    best_user = user_id
            
            # 转换为置信度
            if best_user:
                # 距离越小，置信度越高
                confidence = max(0.0, 1.0 - (best_score / 50.0))
                if confidence > 0.6:
                    return best_user, confidence
        except Exception as e:
            logger.error(f"Failed to recognize voice: {e}")
        
        return None
    
    def _fuse_results(self, face_result: Optional[Tuple[str, float]],
                      voice_result: Optional[Tuple[str, float]]) -> Optional[UserIdentity]:
        """融合人脸识别和声纹识别结果"""
        if not face_result and not voice_result:
            # 陌生人
            return UserIdentity(
                user_id=str(uuid.uuid4())[:8],
                name="陌生人",
                user_type=UserType.STRANGER,
                confidence=0.9,
                recognition_method=RecognitionMethod.BOTH,
                metadata={"reason": "No matching identity found"}
            )
        
        # 只有人脸识别
        if face_result and not voice_result:
            user_id, confidence = face_result
            return UserIdentity(
                user_id=user_id,
                name=user_id,  # 实际应用中应该从用户档案获取
                user_type=UserType.FAMILY,
                confidence=confidence,
                recognition_method=RecognitionMethod.FACE
            )
        
        # 只有声纹识别
        if voice_result and not face_result:
            user_id, confidence = voice_result
            return UserIdentity(
                user_id=user_id,
                name=user_id,  # 实际应用中应该从用户档案获取
                user_type=UserType.FAMILY,
                confidence=confidence,
                recognition_method=RecognitionMethod.VOICE
            )
        
        # 两者都有
        face_user, face_conf = face_result
        voice_user, voice_conf = voice_result
        
        if face_user == voice_user:
            # 结果一致
            confidence = (face_conf + voice_conf) / 2
            return UserIdentity(
                user_id=face_user,
                name=face_user,  # 实际应用中应该从用户档案获取
                user_type=UserType.FAMILY,
                confidence=confidence,
                recognition_method=RecognitionMethod.BOTH
            )
        else:
            # 结果不一致，取置信度高的
            if face_conf > voice_conf:
                return UserIdentity(
                    user_id=face_user,
                    name=face_user,
                    user_type=UserType.FAMILY,
                    confidence=face_conf,
                    recognition_method=RecognitionMethod.FACE
                )
            else:
                return UserIdentity(
                    user_id=voice_user,
                    name=voice_user,
                    user_type=UserType.FAMILY,
                    confidence=voice_conf,
                    recognition_method=RecognitionMethod.VOICE
                )
    
    def is_stranger(self, identity: UserIdentity) -> bool:
        """判断是否是陌生人"""
        return identity.user_type == UserType.STRANGER
    
    def get_user_info(self, user_id: str) -> Dict[str, Any]:
        """获取用户信息"""
        # 实际应用中应该从用户档案系统获取
        return {
            "user_id": user_id,
            "name": user_id,
            "user_type": UserType.FAMILY.value
        }


# 全局身份管理器实例
_global_identity_manager = None


def get_identity_manager() -> IdentityManager:
    """获取身份管理器实例"""
    global _global_identity_manager
    if _global_identity_manager is None:
        _global_identity_manager = IdentityManager()
    return _global_identity_manager


def recognize_user(face_image: Optional[np.ndarray] = None,
                  voice_audio: Optional[np.ndarray] = None,
                  sample_rate: int = 22050) -> Optional[UserIdentity]:
    """识别用户"""
    manager = get_identity_manager()
    return manager.recognize_identity(face_image, voice_audio, sample_rate)


def register_new_user(user_id: str, name: str, face_image: Optional[np.ndarray] = None,
                     voice_audio: Optional[np.ndarray] = None,
                     sample_rate: int = 22050) -> bool:
    """注册新用户"""
    manager = get_identity_manager()
    return manager.register_user(
        user_id=user_id,
        name=name,
        user_type=UserType.FAMILY,
        face_image=face_image,
        voice_audio=voice_audio,
        sample_rate=sample_rate
    )