import tensorflow as tf
import numpy as np


# Создание модели шумоподавления
def create_noise_reduction_model():
    # Определяем входные слои
    input_audio = tf.keras.layers.Input(shape=(None,), name='audio_input')
    input_noise = tf.keras.layers.Input(shape=(None,), name='noise_profile')

    # STFT в виде слоев
    def stft_layer(x):
        stft = tf.signal.stft(x, frame_length=512, frame_step=128)
        magnitude = tf.abs(stft)
        phase = tf.math.angle(stft)
        return magnitude, phase

    audio_mag, audio_phase = tf.keras.layers.Lambda(lambda x: stft_layer(x))(input_audio)
    noise_mag, _ = tf.keras.layers.Lambda(lambda x: stft_layer(x))(input_noise)

    # Конкатенация спектров
    concat_input = tf.keras.layers.Concatenate()([audio_mag, noise_mag])

    x = tf.keras.layers.Reshape((-1, concat_input.shape[-1]))(concat_input)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    mask = tf.keras.layers.Dense(257, activation='sigmoid')(x)

    # Применение маски
    enhanced_mag = tf.keras.layers.Multiply()([audio_mag, mask])

    # Собираем комплексный спектр и применяем ISTFT
    def inverse_stft_layer(inputs):
        mag, phase = inputs
        complex_stft = tf.complex(mag * tf.math.cos(phase), mag * tf.math.sin(phase))
        return tf.signal.inverse_stft(complex_stft, frame_length=512, frame_step=128)

    enhanced_audio = tf.keras.layers.Lambda(inverse_stft_layer)([enhanced_mag, audio_phase])

    model = tf.keras.Model(inputs=[input_audio, input_noise], outputs=enhanced_audio)
    model.compile(optimizer='adam', loss='mse')

    print("✅ Модель шумоподавления создана")

    
    # Конвертация в TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # ⚠️ Включаем поддержку нестандартных операций и отключаем TensorList lowering
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.target_spec.supported_versions = [1, 2]
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()

    with open('../assets/models/noise_reduction_model.tflite', 'wb') as f:
        f.write(tflite_model)
        print("✅ Модель шумоподавления сохранена")


# Создание модели распознавания речи
def create_speech_recognition_model():
    input_audio = tf.keras.layers.Input(shape=(16000,), name='audio_input')
    input_accent = tf.keras.layers.Input(shape=(64,), name='accent_profile')

    def extract_log_mel_mfcc(x):
        stft = tf.signal.stft(x, frame_length=512, frame_step=256)
        spectrogram = tf.abs(stft)

        num_mel_bins = 40
        mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_mel_bins,
            num_spectrogram_bins=stft.shape[-1],
            sample_rate=16000,
            lower_edge_hertz=20,
            upper_edge_hertz=8000
        )

        mel_spectrogram = tf.tensordot(spectrogram, mel_weight_matrix, 1)
        mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate([num_mel_bins]))
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
        return log_mel_spectrogram

    mfccs = tf.keras.layers.Lambda(extract_log_mel_mfcc)(input_audio)
    mfccs = tf.keras.layers.LayerNormalization()(mfccs)

    x = tf.keras.layers.Reshape((-1, 40, 1))(mfccs)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Concatenate()([x, input_accent])
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    num_commands = 6
    output = tf.keras.layers.Dense(num_commands, activation='softmax')(x)

    model = tf.keras.Model(inputs=[input_audio, input_accent], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("✅ Модель распознавания речи создана")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # ⚠️ Включаем поддержку нестандартных операций и отключаем TensorList lowering
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()

    with open('../assets/models/speech_recognition_model.tflite', 'wb') as f:
        f.write(tflite_model)
        print("✅ Модель распознавания речи сохранена")


if __name__ == "__main__":
    print("🔄️ Создание моделей TensorFlow Lite для Flutter-приложения...")
    create_noise_reduction_model()
    create_speech_recognition_model()
    print("✅ Обе модели созданы и сохранены")
