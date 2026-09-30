# Edge AI Projects

## A. Sensing and foundations

1. Raspberry Pi 5 DHT11/DHT22 sensing — GPIO wiring, polling, errors, live trace.
2. Raspberry Pi 5 Flask weather station — Open-Meteo, JSON, local dashboard, network failure.
3. Raspberry Pi 5 Blynk dashboard — virtual pins, sensor history, sustained alerts.
4. Factory defect detection — edge versus cloud latency, bandwidth, privacy, TFLite.
5. Delivery drone hardware selection — STM32/Pi/Jetson, mean and tail latency, energy.
6. Vaccine cold-chain monitoring — MQTT, temperature/vibration fusion, TFLite, SQLite, alerts.

## B. TinyML and board deployment

7. First TinyML pipeline — Edge Impulse and code-first paths.
8. Keyword spotting — Nano 33 BLE Sense.
9. Classification and confusion matrix — Nano/ESP32.
10. IMU “Magic Wand” gesture recognition — Nano 33 BLE Sense.
11. Sensor anomaly detection — ESP32-S3.
12. MCU deployment — STM32, TFLite Micro, CMSIS-NN, tensor arena.
13. Custom object detection — Raspberry Pi/Jetson.
14. MobileNetV1/V2 transfer learning — Raspberry Pi/Jetson.
15. Edge TPU compilation and CPU fallback — Coral.

## C. Vision, acceleration, and runtimes

16. Image, video, and optical flow — Raspberry Pi/Jetson.
17. Jetson vision tasks — classification, detection, segmentation, pose.
18. Local video redaction and privacy — Raspberry Pi/Jetson.
19. NumPy versus CuPy/Numba/PyCUDA — Jetson GPU benchmark.
20. ONNX export and graph inspection.
21. ONNX Runtime execution providers and fallback.
22. **Face detection on Raspberry Pi 5 with OpenVINO CPU** — camera input, model I/O, bounding boxes, latency, and FPS. No Intel Neural Compute Stick is used.
23. TensorRT deployment — Jetson Orin, FP32/FP16/INT8.

## D. Optimization, lifecycle, and trust

24. Quantization and smart-wristband constraint — FP32/FP16/INT8, PTQ/QAT, calibration, Pareto choice.
25. Pruning, distillation, and operator fusion.
26. Traffic-camera model lifecycle — conversion, update, monitoring, rollback.
27. Retail-camera Edge MLOps — registry, OTA, drift, shadow deployment, rollback.
28. Federated hospital training — FedAvg, local data, leakage, trust.

## E. Robotics and advanced Edge AI

29. ROS 2 perception-to-action pipeline — Jetson/JetBot.
30. Robot navigation — JetBot with Gazebo/Isaac Sim.
31. DQN agent — simulation.
32. SLM memory and KV-cache lab — UNO Q/Pi/Jetson.
33. One-bit embeddings and local RAG — UNO Q.
34. Privacy-first voice kiosk — UNO Q 4 GB, Whisper/Moorcheh/Llama 3.2 1B/Piper.
35. Training-free food understanding — ingredients, caption reranking, calorie estimation.
36. Sensor tokenization — STWIN.box residual VQ and 250 symbols.
37. Six-board industrial monitoring — reference versus five targets, replay, triage.
38. Cross-device deployment and production checks — Jetson AGX Orin/IQ-9075, hashes, schema, fail-closed behavior.

## Completion and freeze gate

1. Board-specific workspace and correct wiring/interfaces where applicable.
2. Step-by-step interactive run with adjustable inputs, intermediate states, and failure cases.
3. Runnable code, setup instructions, sample data, and expected output.
4. Verified calculations and clear labels for simulation versus measured results.
5. Student task, assessment questions, and reproducible result sheet.

Sources: [OpenVINO installation](https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-pip.html), [Open Model Zoo model](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/face-detection-retail-0004), [Raspberry Pi camera software](https://www.raspberrypi.com/documentation/computers/camera_software.html).
