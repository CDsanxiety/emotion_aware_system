# edge_cloud_orchestrator.py
"""
边缘-云端协同编排模块 (Edge-Cloud Orchestration)
定义感知边缘化与认知云端化的边界
实现动态资源调度：网络自适应降级/升级
"""
import time
import threading
import subprocess
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import queue


class TaskType(Enum):
    EDGE = "edge"
    CLOUD = "cloud"
    HYBRID = "hybrid"


class TaskCategory(Enum):
    PERCEPTION_RAW = "perception_raw"
    PERCEPTION_EMOTION = "perception_emotion"
    PERCEPTION_SCENE = "perception_scene"
    COGNITION_REASONING = "cognition_reasoning"
    COGNITION_DECISION = "cognition_decision"
    COGNITION_PLANNING = "cognition_planning"
    EXECUTION_CONTROL = "execution_control"
    MEMORY_STORAGE = "memory_storage"
    MEMORY_RETRIEVAL = "memory_retrieval"


class NetworkStatus(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    OFFLINE = "offline"


class ExecutionMode(Enum):
    FULL_CLOUD = "full_cloud"
    EDGE_PREFERRED = "edge_preferred"
    CLOUD_PREFERRED = "cloud_preferred"
    FULL_EDGE = "full_edge"
    ADAPTIVE = "adaptive"


@dataclass
class OrchestrationTask:
    task_id: str
    task_type: TaskType
    category: TaskCategory
    data: Dict[str, Any]
    priority: int = 5
    timeout: float = 10.0
    fallback_enabled: bool = True
    created_at: float = 0.0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


@dataclass
class ExecutionResult:
    task_id: str
    success: bool
    result: Any
    execution_location: str
    execution_time: float
    error: Optional[str] = None
    fallback_used: bool = False


class NetworkMonitor:
    def __init__(self):
        self._status = NetworkStatus.GOOD
        self._bandwidth_mbps: float = 100.0
        self._latency_ms: float = 50.0
        self._packet_loss: float = 0.0
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_check = 0.0
        self._check_interval = 5.0

    def start_monitoring(self) -> None:
        if self._monitoring:
            return
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

    def _monitor_loop(self) -> None:
        while self._monitoring:
            self._check_network()
            time.sleep(self._check_interval)

    def _check_network(self) -> None:
        try:
            start = time.time()
            result = subprocess.run(
                ["ping", "-n", "1", "-w", "1000", "8.8.8.8"],
                capture_output=True,
                text=True,
                timeout=2
            )
            elapsed = (time.time() - start) * 1000

            with self._lock:
                if result.returncode == 0:
                    self._latency_ms = elapsed
                    self._packet_loss = 0.0
                    if elapsed < 50:
                        self._status = NetworkStatus.EXCELLENT
                    elif elapsed < 100:
                        self._status = NetworkStatus.GOOD
                    elif elapsed < 300:
                        self._status = NetworkStatus.FAIR
                    else:
                        self._status = NetworkStatus.POOR
                else:
                    self._status = NetworkStatus.OFFLINE
                    self._packet_loss = 100.0

            self._last_check = time.time()
        except Exception:
            with self._lock:
                self._status = NetworkStatus.POOR

    def force_check(self) -> NetworkStatus:
        self._check_network()
        return self.get_status()

    def get_status(self) -> NetworkStatus:
        with self._lock:
            return self._status

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "status": self._status.value,
                "latency_ms": self._latency_ms,
                "bandwidth_mbps": self._bandwidth_mbps,
                "packet_loss": self._packet_loss,
                "last_check": self._last_check
            }

    def is_cloud_available(self) -> bool:
        return self.get_status() not in [NetworkStatus.POOR, NetworkStatus.OFFLINE]

    def should_use_cloud(self, task_category: TaskCategory) -> bool:
        status = self.get_status()

        if status == NetworkStatus.OFFLINE:
            return False
        if status == NetworkStatus.POOR:
            return task_category in [TaskCategory.COGNITION_REASONING, TaskCategory.COGNITION_DECISION]
        if status == NetworkStatus.EXCELLENT:
            return True
        if status == NetworkStatus.GOOD:
            return task_category not in [TaskCategory.PERCEPTION_RAW]
        if status == NetworkStatus.FAIR:
            return task_category in [TaskCategory.COGNITION_REASONING, TaskCategory.COGNITION_DECISION]

        return False


class TaskBoundary:
    EDGE_TASKS = {
        TaskCategory.PERCEPTION_RAW,
        TaskCategory.EXECUTION_CONTROL,
    }

    CLOUD_TASKS = {
        TaskCategory.COGNITION_REASONING,
        TaskCategory.COGNITION_DECISION,
        TaskCategory.COGNITION_PLANNING,
    }

    HYBRID_TASKS = {
        TaskCategory.PERCEPTION_EMOTION,
        TaskCategory.PERCEPTION_SCENE,
        TaskCategory.MEMORY_STORAGE,
        TaskCategory.MEMORY_RETRIEVAL,
    }

    @classmethod
    def get_task_type(cls, category: TaskCategory) -> TaskType:
        if category in cls.EDGE_TASKS:
            return TaskType.EDGE
        elif category in cls.CLOUD_TASKS:
            return TaskType.CLOUD
        elif category in cls.HYBRID_TASKS:
            return TaskType.HYBRID
        return TaskType.HYBRID

    @classmethod
    def get_timeout(cls, category: TaskCategory, is_cloud: bool) -> float:
        base_timeouts = {
            TaskCategory.PERCEPTION_RAW: 0.1,
            TaskCategory.PERCEPTION_EMOTION: 0.5,
            TaskCategory.PERCEPTION_SCENE: 1.0,
            TaskCategory.COGNITION_REASONING: 5.0,
            TaskCategory.COGNITION_DECISION: 3.0,
            TaskCategory.COGNITION_PLANNING: 10.0,
            TaskCategory.EXECUTION_CONTROL: 0.5,
            TaskCategory.MEMORY_STORAGE: 1.0,
            TaskCategory.MEMORY_RETRIEVAL: 2.0,
        }
        base = base_timeouts.get(category, 5.0)
        return base * 1.5 if not is_cloud else base


class ResourceScheduler:
    def __init__(self, network_monitor: NetworkMonitor):
        self._network_monitor = network_monitor
        self._execution_mode = ExecutionMode.ADAPTIVE
        self._custom_rules: Dict[str, Callable] = {}
        self._task_handlers: Dict[TaskCategory, Dict[TaskType, Callable]] = defaultdict(dict)
        self._lock = threading.Lock()

    def set_execution_mode(self, mode: ExecutionMode) -> None:
        with self._lock:
            self._execution_mode = mode

    def get_execution_mode(self) -> ExecutionMode:
        with self._lock:
            return self._execution_mode

    def register_handler(self, category: TaskCategory, task_type: TaskType,
                        handler: Callable) -> None:
        self._task_handlers[category][task_type] = handler

    def register_custom_rule(self, rule_name: str, rule_func: Callable) -> None:
        self._custom_rules[rule_name] = rule_func

    def should_execute_on_cloud(self, task: OrchestrationTask) -> bool:
        if self._execution_mode == ExecutionMode.FULL_CLOUD:
            return True
        elif self._execution_mode == ExecutionMode.FULL_EDGE:
            return False
        elif self._execution_mode == ExecutionMode.CLOUD_PREFERRED:
            return self._network_monitor.is_cloud_available()
        elif self._execution_mode == ExecutionMode.EDGE_PREFERRED:
            return False

        return self._network_monitor.should_use_cloud(task.category)

    def get_handler(self, task: OrchestrationTask) -> tuple[Callable, TaskType]:
        preferred_cloud = self.should_execute_on_cloud(task)
        task_type = TaskType.CLOUD if preferred_cloud else TaskType.EDGE

        if task_type in self._task_handlers[task.category]:
            return self._task_handlers[task.category][task_type], task_type

        if TaskType.HYBRID in self._task_handlers[task.category]:
            return self._task_handlers[task.category][TaskType.HYBRID], TaskType.HYBRID

        return None, task_type

    def get_fallback_handler(self, task: OrchestrationTask) -> tuple[Callable, TaskType]:
        preferred_cloud = self.should_execute_on_cloud(task)
        fallback_cloud = not preferred_cloud
        fallback_type = TaskType.CLOUD if fallback_cloud else TaskType.EDGE

        if fallback_type in self._task_handlers[task.category]:
            return self._task_handlers[task.category][fallback_type], fallback_type

        if TaskType.HYBRID in self._task_handlers[task.category]:
            return self._task_handlers[task.category][TaskType.HYBRID], TaskType.HYBRID

        return None, fallback_type


class EdgeCloudOrchestrator:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._network_monitor = NetworkMonitor()
        self._scheduler = ResourceScheduler(self._network_monitor)
        self._execution_history: List[ExecutionResult] = []
        self._max_history = 100
        self._lock_history = threading.Lock()
        self._task_queue: queue.Queue = queue.Queue()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._execution_stats: Dict[str, int] = defaultdict(int)

    @classmethod
    def get_instance(cls) -> "EdgeCloudOrchestrator":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def initialize(self) -> None:
        self._network_monitor.start_monitoring()
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def shutdown(self) -> None:
        self._running = False
        self._network_monitor.stop_monitoring()
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)

    def _worker_loop(self) -> None:
        while self._running:
            try:
                task = self._task_queue.get(timeout=1.0)
                self._process_task(task)
            except queue.Empty:
                continue

    def _process_task(self, task: OrchestrationTask) -> None:
        handler, execution_location = self._scheduler.get_handler(task)

        if not handler:
            result = ExecutionResult(
                task_id=task.task_id,
                success=False,
                result=None,
                execution_location="none",
                execution_time=0.0,
                error=f"No handler for {task.category}"
            )
            self._add_result(result)
            return

        start_time = time.time()
        fallback_used = False

        try:
            result = handler(task.data)
        except Exception as e:
            if task.fallback_enabled:
                fallback_handler, fallback_location = self._scheduler.get_fallback_handler(task)
                if fallback_handler:
                    try:
                        result = fallback_handler(task.data)
                        fallback_used = True
                    except Exception:
                        self._record_failure(task, execution_location, start_time, str(e))
                        return
                else:
                    self._record_failure(task, execution_location, start_time, str(e))
                    return
            else:
                self._record_failure(task, execution_location, start_time, str(e))
                return

        execution_time = time.time() - start_time
        exec_result = ExecutionResult(
            task_id=task.task_id,
            success=True,
            result=result,
            execution_location=execution_location.value,
            execution_time=execution_time,
            fallback_used=fallback_used
        )
        self._add_result(exec_result)

    def _record_failure(self, task: OrchestrationTask, execution_location: TaskType,
                        start_time: float, error: str) -> None:
        execution_time = time.time() - start_time
        exec_result = ExecutionResult(
            task_id=task.task_id,
            success=False,
            result=None,
            execution_location=execution_location.value,
            execution_time=execution_time,
            error=error
        )
        self._add_result(exec_result)

    def submit_task(self, task: OrchestrationTask) -> str:
        self._task_queue.put(task)
        return task.task_id

    def execute_task_sync(self, task: OrchestrationTask) -> ExecutionResult:
        handler, execution_location = self._scheduler.get_handler(task)

        if not handler:
            return ExecutionResult(
                task_id=task.task_id,
                success=False,
                result=None,
                execution_location="none",
                execution_time=0.0,
                error=f"No handler for {task.category}"
            )

        start_time = time.time()
        fallback_used = False

        try:
            result = handler(task.data)
        except Exception as e:
            if task.fallback_enabled:
                fallback_handler, fallback_location = self._scheduler.get_fallback_handler(task)
                if fallback_handler:
                    try:
                        result = fallback_handler(task.data)
                        fallback_used = True
                    except Exception:
                        return self._record_failure(task, execution_location, start_time, str(e))
                else:
                    return self._record_failure(task, execution_location, start_time, str(e))
            else:
                return self._record_failure(task, execution_location, start_time, str(e))

        execution_time = time.time() - start_time
        exec_result = ExecutionResult(
            task_id=task.task_id,
            success=True,
            result=result,
            execution_location=execution_location.value,
            execution_time=execution_time,
            fallback_used=fallback_used
        )
        self._add_result(exec_result)
        return exec_result

    def _add_result(self, result: ExecutionResult) -> None:
        with self._lock_history:
            self._execution_history.append(result)
            if len(self._execution_history) > self._max_history:
                self._execution_history.pop(0)

        self._execution_stats[result.execution_location] += 1

    def set_execution_mode(self, mode: ExecutionMode) -> None:
        self._scheduler.set_execution_mode(mode)

    def get_execution_mode(self) -> ExecutionMode:
        return self._scheduler.get_execution_mode()

    def get_network_status(self) -> NetworkStatus:
        return self._network_monitor.get_status()

    def get_network_metrics(self) -> Dict[str, Any]:
        return self._network_monitor.get_metrics()

    def get_execution_stats(self) -> Dict[str, Any]:
        return {
            "total_executions": sum(self._execution_stats.values()),
            "by_location": dict(self._execution_stats),
            "recent_results": [r.task_id for r in self._execution_history[-5:]]
        }

    def get_execution_history(self, limit: int = 10) -> List[ExecutionResult]:
        with self._lock_history:
            return self._execution_history[-limit:]

    def register_task_handler(self, category: TaskCategory, task_type: TaskType,
                             handler: Callable) -> None:
        self._scheduler.register_handler(category, task_type, handler)

    def force_network_check(self) -> NetworkStatus:
        return self._network_monitor.force_check()

    def get_task_recommendation(self, category: TaskCategory) -> Dict[str, Any]:
        network_status = self.get_network_status()
        recommended_cloud = self._network_monitor.should_use_cloud(category)

        return {
            "task_category": category.value,
            "network_status": network_status.value,
            "recommended_location": "cloud" if recommended_cloud else "edge",
            "fallback_location": "edge" if recommended_cloud else "cloud",
            "task_type": TaskBoundary.get_task_type(category).value
        }


_global_orchestrator: Optional[EdgeCloudOrchestrator] = None


def get_orchestrator() -> EdgeCloudOrchestrator:
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = EdgeCloudOrchestrator.get_instance()
    return _global_orchestrator


def is_cloud_available() -> bool:
    orchestrator = get_orchestrator()
    return orchestrator.get_network_status() not in [NetworkStatus.POOR, NetworkStatus.OFFLINE]


def get_task_recommendation(category: TaskCategory) -> Dict[str, Any]:
    orchestrator = get_orchestrator()
    return orchestrator.get_task_recommendation(category)
