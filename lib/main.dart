import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'pages/voice_assistant_page.dart';
import 'services/audio_service.dart';
import 'services/tflite_service.dart';

final GlobalKey<NavigatorState> navigatorKey = GlobalKey<NavigatorState>();

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const VoiceAssistantApp());
}

class VoiceAssistantApp extends StatelessWidget {
  const VoiceAssistantApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => AudioService()),
        ChangeNotifierProvider(create: (_) => TFLiteService()),
      ],
      child: MaterialApp(
        navigatorKey: navigatorKey,
        debugShowCheckedModeBanner: false,
        title: 'Речевой ассистент',
        theme: ThemeData(primarySwatch: Colors.blue),
        home: const VoiceAssistantPage(),
      ),
    );
  }
}
