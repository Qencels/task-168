import logging

import numpy as np
import torch

import config as cfg


def tts_init():
    try:
        logging.debug(f"Загрузка Silero TTS '{cfg.TTS_MODEL_VERSION}'...")
        silero_tts_model, _ = torch.hub.load(
            repo_or_dir=cfg.TTS_SILERO_REPO,
            model=cfg.TTS_SILERO_MODEL,
            language=cfg.LANGUAGE,
            speaker=cfg.TTS_MODEL_VERSION
        )
        logging.debug(f"Модель Silero TTS '{cfg.TTS_MODEL_VERSION}' загружена.")

    except Exception as e:
        raise
    return silero_tts_model


def generate_audio_data(tts_model):
    logging.info(f"Генерация аудио для прогрева моделей...")

    audio_data_list = []

    for i in range(cfg.TTS_NUM_SPEAKERS):
        try:
            audio_segment = tts_model.apply_tts(
                text=cfg.TTS_WARMUP_TEXT,
                speaker=cfg.TTS_SPEAKER,
                sample_rate=cfg.TTS_SAMPLE_RATE
            ).numpy()
            audio_data_list.append(audio_segment)

        except Exception as e:
            logging.warning(f"Ошибка при синтезе аудио: {e}")

    if audio_data_list:
        audio_data = np.concatenate(audio_data_list)
    else:

        logging.warning("Не удалось сгенерировать аудио ни одним спикером. Используется тишина для прогрева.")
        audio_data = np.zeros(int(cfg.TTS_SAMPLE_RATE * 1.0), dtype=np.float32)

    logging.info("Аудио для прогрева сгенерировано.")
    return audio_data


def warmup_models(whisper_model, vad_model, timestamps_func):
    try:
        logging.info("Прогрев моделей Whisper и Silero VAD...")
        tts_model = tts_init()
        warmup_audio_data = generate_audio_data(tts_model)

        for i in range(cfg.NUM_WARMUP_RUNS):
            _ = whisper_model.transcribe(warmup_audio_data, language=cfg.LANGUAGE)
            _ = timestamps_func(warmup_audio_data, vad_model, sampling_rate=cfg.SAMPLE_RATE, **cfg.VAD_PARAMETERS)
            logging.debug(f"Прогрев: запуск {i + 1}/{cfg.NUM_WARMUP_RUNS} завершен.")

        logging.info("Прогрев моделей завершен.")

    except Exception as e:
        raise
