import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/audio_service.dart';
import '../services/tflite_service.dart';

class VoiceAssistantPage extends StatelessWidget {
  const VoiceAssistantPage({super.key});

  @override
  Widget build(BuildContext context) {
    final audioService = Provider.of<AudioService>(context);
    final tfliteService = Provider.of<TFLiteService>(context);

    return Scaffold(
      appBar: AppBar(title: const Text('Голосовой ассистент')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Text('Распознанный текст: \n ${tfliteService.resultText}'),
            const SizedBox(height: 20),
            Text('Статус: ${audioService.isRecording ? "Запись..." : "Ожидание"}'),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () => audioService.toggleRecording(),
              child: Text(audioService.isRecording ? 'Остановить' : 'Начать запись'),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () => audioService.recordAccentSample(),
              child: const Text('Обновить акцент'),
            ),
            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }
}
