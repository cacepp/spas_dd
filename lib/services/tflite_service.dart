import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import '../models/accent_profile.dart';

class TFLiteService with ChangeNotifier {
  late Interpreter _noiseReductionInterpreter;
  late Interpreter _speechRecognitionInterpreter;

  AccentProfile _accentProfile = AccentProfile(List.filled(64, 0.0));

  String resultText = '';

  TFLiteService() {
    _loadModels();
    _loadAccentProfile();
  }

  final options = InterpreterOptions()..useNnApiForAndroid = true;

  Future<void> _loadModels() async {
    try {
      _noiseReductionInterpreter = await Interpreter.fromAsset(
        'assets/models/noise_reduction_model.tflite',
        options: options,
      );

      _speechRecognitionInterpreter = await Interpreter.fromAsset(
        'assets/models/speech_recognition_model.tflite',
        options: options,
      );

      debugPrint('✅ Модели загружены');
    } catch (e) {
      debugPrint('❌ Ошибка загрузки моделей: $e');
    }
  }

  Future<void> _loadAccentProfile() async {
    final prefs = await SharedPreferences.getInstance();
    final stored = prefs.getStringList('accent_profile');
    if (stored != null) {
      _accentProfile = AccentProfile(stored.map((e) => double.tryParse(e) ?? 0.0).toList());
    }
  }

  Future<void> saveAccentProfile(AccentProfile profile) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList(
      'accent_profile',
      profile.vector.map((e) => e.toString()).toList(),
    );
    _accentProfile = profile;
  }

  AccentProfile getAccentProfile() => _accentProfile;

  List<double> _runNoiseReduction(List<double> audio, List<double> noise) {
    final inputAudio = [audio];
    final inputNoise = [noise];

    final output = List.generate(1, (_) => List.filled(audio.length, 0.0));

    _noiseReductionInterpreter.runForMultipleInputs([inputAudio, inputNoise], [output] as Map<int, Object>);
    return output.first;
  }

  String _runSpeechRecognition(List<double> audio, List<double> accent) {
    final inputAudio = [audio];
    final inputAccent = [accent];

    final output = List.filled(1 * 6, 0.0).reshape([1, 6]);

    _speechRecognitionInterpreter.runForMultipleInputs([inputAudio, inputAccent], [output] as Map<int, Object>);

    final scores = output.first;
    final maxScore = scores.reduce((a, b) => a > b ? a : b);
    final commandIndex = scores.indexOf(maxScore);

    const commands = ['открыть', 'закрыть', 'позвонить', 'сообщение', 'музыка', 'погода'];
    return commands[commandIndex];
  }

  Future<void> runInference({
    required List<double> audio,
    required List<double> noiseProfile,
  }) async {
    try {
      final denoised = _runNoiseReduction(audio, noiseProfile);
      final command = _runSpeechRecognition(denoised, _accentProfile.vector);
      resultText = command;
      notifyListeners();
    } catch (e) {
      debugPrint('❌ Ошибка инференса: $e');
    }
  }

  List<double> _extractSimpleMFCC(List<double> audio) {
    const frameSize = 512;
    final frames = <List<double>>[];

    for (int i = 0; i + frameSize < audio.length; i += frameSize) {
      frames.add(audio.sublist(i, i + frameSize));
    }

    final mfccs = frames.map((f) {
      final avg = f.reduce((a, b) => a + b) / f.length;
      return List.filled(4, avg); // грубая эмбеддинговая заглушка
    }).expand((e) => e).toList();

    return mfccs.length >= 64
        ? mfccs.sublist(0, 64)
        : [...mfccs, ...List.filled(64 - mfccs.length, 0.0)];
  }

  Future<void> generateAndSaveAccentProfile(List<double> audioSample) async {
    final mfccVector = _extractSimpleMFCC(audioSample);
    final profile = AccentProfile(mfccVector);
    await saveAccentProfile(profile);
    debugPrint('✅ Профиль акцента сохранён');
  }
}
