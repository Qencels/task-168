import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')

logging.info(f"Скрипт {os.path.basename(__file__)} запущен и ожидает бинарные данные на стандартном вводе.")

BLOCK_SIZE = 4096

try:
    while True:
        data = sys.stdin.buffer.read(BLOCK_SIZE)
        if not data:
            logging.info("Получен сигнал конца стандартного ввода (EOF). Завершение работы.")
            break
        logging.info(f"Получено {len(data)} байт бинарных данных.")

except Exception as e:
    logging.error(f"Произошла ошибка при чтении из стандартного ввода: {e}")
except KeyboardInterrupt:
    logging.info("Получен сигнал прерывания (Ctrl+C). Завершение работы.")
finally:
    logging.info(f"Скрипт {os.path.basename(__file__)} завершает работу.")
    sys.stdout.flush()
