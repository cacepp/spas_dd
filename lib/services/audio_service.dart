import 'package:flutter/material.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:provider/provider.dart';
import 'package:spas_dd/services/tflite_service.dart';
import 'dart:typed_data';
import 'dart:async';
import 'dart:typed_data';

import '../main.dart';

class AudioService with ChangeNotifier {
  final FlutterSoundRecorder _recorder = FlutterSoundRecorder();
  bool isRecording = false;
  final List<double> _recordedSamples = [];
  final List<double> _noiseSamples = [];
  StreamController<Uint8List>? _audioStreamController;

  AudioService() {
    _initRecorder();
  }

  Future<void> _initRecorder() async {
    await Permission.microphone.request();
    await _recorder.openRecorder();
    _recorder.setSubscriptionDuration(const Duration(milliseconds: 100));
  }

  void toggleRecording() {
    isRecording ? stopRecording() : startRecording();
  }

  Future<void> startRecording() async {
    if (!await Permission.microphone.isGranted) {
      debugPrint('❌ Нет доступа к микрофону');
      return;
    }

    _recordedSamples.clear();
    isRecording = true;
    notifyListeners();

    // Создаем контроллер потока для аудио данных
    _audioStreamController = StreamController<Uint8List>();
    _audioStreamController!.stream.listen((buffer) {
      final samples = _convertToFloat32(buffer);
      _recordedSamples.addAll(samples);
    });

    await _recorder.startRecorder(
      codec: Codec.pcm16,
      sampleRate: 16000,
      numChannels: 1,
      toStream: _audioStreamController!.sink,
    );
  }

  Future<void> stopRecording() async {
    await _recorder.stopRecorder();

    // Закрываем контроллер потока
    await _audioStreamController?.close();
    _audioStreamController = null;

    isRecording = false;
    notifyListeners();

    final audio = _normalize(_recordedSamples);

    // Примерно первые 0.5 сек — шум
    final noise = audio.take(8000).toList();
    final command = audio.skip(8000).toList();

    // Выполнить инференс
    await runInference(audioData: command, noiseProfile: noise);
  }

  List<double> _convertToFloat32(Uint8List buffer) {
    final int16buffer = Int16List.view(buffer.buffer);
    return int16buffer.map((e) => e / 32768.0).toList();
  }

  List<double> _normalize(List<double> data) {
    final maxVal = data.fold<double>(0.0, (a, b) => a.abs() > b.abs() ? a : b).abs();
    return maxVal == 0.0 ? data : data.map((e) => e / maxVal).toList();
  }

  Future<void> runInference({
    required List<double> audioData,
    required List<double> noiseProfile,
  }) async {
    final context = navigatorKey.currentContext!;
    final tflite = Provider.of<TFLiteService>(context, listen: false);
    await tflite.runInference(audio: audioData, noiseProfile: noiseProfile);
  }

  void dispose() {
    _audioStreamController?.close();
    _recorder.closeRecorder();
    super.dispose();
  }

  Future<void> recordAccentSample() async {
    if (!await Permission.microphone.isGranted) {
      debugPrint('❌ Нет доступа к микрофону');
      return;
    }

    _recordedSamples.clear();

    // Создаем контроллер потока для аудио данных
    _audioStreamController = StreamController<Uint8List>();
    _audioStreamController!.stream.listen((buffer) {
      final samples = _convertToFloat32(buffer);
      _recordedSamples.addAll(samples);
    });

    await _recorder.startRecorder(
      codec: Codec.pcm16,
      sampleRate: 16000,
      numChannels: 1,
      toStream: _audioStreamController!.sink,
    );

    // Записываем 1.5 сек, затем останавливаем
    await Future.delayed(const Duration(milliseconds: 1500));
    await _recorder.stopRecorder();

    // Закрываем контроллер потока
    await _audioStreamController?.close();
    _audioStreamController = null;

    final audio = _normalize(_recordedSamples);
    final context = navigatorKey.currentContext!;
    final tflite = Provider.of<TFLiteService>(context, listen: false);
    await tflite.generateAndSaveAccentProfile(audio);

    notifyListeners();
  }
}
