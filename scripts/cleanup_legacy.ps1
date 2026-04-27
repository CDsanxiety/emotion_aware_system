# scripts/cleanup_legacy.sh (Actually PowerShell because we are on Windows)
# Use with caution!
# Remove redundant files after refactoring
Remove-Item -Path "app.py", "agent_loop.py", "audio.py", "audio_manager.py", "audio_player_driver.py", "autogen_integration.py", "blackboard.py", "config.py", "decision_tracer.py", "download_assets.py", "edge_cloud_orchestrator.py", "identity_manager.py", "identity_registration_agent.py", "led_hardware_driver.py", "llm_api.py", "memory_rag.py", "multi_agent.py", "openvla_integration.py", "pad_model.py", "physical_expression.py", "ros_bridge.py", "ros_client.py", "safety_guardrails.py", "social_norms.py", "tts.py", "uncertainty.py", "utils.py", "vision.py" -ErrorAction SilentlyContinue
Remove-Item -Path "v2", "workspace", ".idea", ".agents" -Recurse -ErrorAction SilentlyContinue
Write-Host "Legacy files removed. Project is now clean."
