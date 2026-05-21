import serial.tools
import serial.tools.list_ports
from transformers import AutoModelForSequenceClassification, AutoTokenizer, WhisperProcessor
import sounddevice as sd
import scipy.io.wavfile as wav
import noisereduce as nr
import numpy as np
import speech_recognition as sr
from dotenv import load_dotenv
from vosk import Model, KaldiRecognizer
from pywifi import PyWiFi, const, Profile
import ctranslate2, torch, time, pyaudio, socket, pygame, os, requests, serial, json, sys
import threading

# --- Global State and Configuration ---
program_state = {
    "last_command" : "nyalakan_lampu", # Initialize with a common command
    "current_light_state" : "ON",
    "gender" : "pria",
    "predicted_language_from_wake_word": "Indonesian",
    "last_command_for_feedback_purpose": None # Used by intent_feedback
}

ESP = None
USING_ESP = False # Set True jika ingin menggunakan ESP32

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("PERINGATAN: GEMINI_API_KEY tidak ditemukan di .env file. Fungsi online intent akan gagal.")
    GEMINI_URL = ""
else:
    GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=" + GEMINI_API_KEY

try:
    pygame.mixer.init()
except pygame.error as e:
    print(f"Gagal inisialisasi pygame mixer: {e}. Feedback suara mungkin tidak berfungsi.")

this_file_dir = os.path.dirname(os.path.abspath(__file__))

# Inisialisasi model NLP
try:
    nlp_model_path = os.path.join(this_file_dir, "natural_language_processing/mdeberta-intent-classification-final")
    nlp_tokenizer = AutoTokenizer.from_pretrained(nlp_model_path)
    nlp_model = AutoModelForSequenceClassification.from_pretrained(nlp_model_path)
    nlp_model = nlp_model.eval().to("cpu")
    nlp_label_list = [
        "nyalakan lampu merah", "nyalakan lampu hijau", "nyalakan lampu biru",
        "nyalakan lampu lavender", "nyalakan lampu magenta", "nyalakan lampu pink",
        "nyalakan lampu violet", "nyalakan lampu aqua", "nyalakan lampu kuning",
        "nyalakan lampu emas", "nyalakan lampu abu", "nyalakan mode senang",
        "nyalakan mode sad", "matikan lampu", "tidak relevan", "nyalakan lampu",
        "gender ke wanita", "gender ke pria"
    ]
    id_to_label = {idx: label for idx, label in enumerate(nlp_label_list)}
except Exception as e:
    print(f"Gagal memuat model NLP: {e}. Program mungkin tidak berfungsi dengan benar.")
    sys.exit(1)

# Inisialisasi model STT
try:
    stt_processor = WhisperProcessor.from_pretrained(os.path.join(this_file_dir, "speech_to_text/whisper_finetuned"))
    stt_model = ctranslate2.models.Whisper(os.path.join(this_file_dir, "speech_to_text/whisper_finetuned_ct2"))
except Exception as e:
    print(f"Gagal memuat model STT: {e}. Program mungkin tidak berfungsi dengan benar.")
    sys.exit(1)

# Inisialisasi Vosk (Model loaded later in main)
wake_model = None

# --- Ring Buffer Implementation ---
BUFFER_SIZE_BYTES = 300000
audio_ring_buffer = bytearray(BUFFER_SIZE_BYTES)
write_pos = 0
buffer_lock = threading.Lock()
data_available = threading.Condition(buffer_lock)
total_written_bytes = 0

def callback(indata, frames, time_info, status):
    global write_pos, total_written_bytes, audio_ring_buffer, buffer_lock, data_available
    if status:
        if status.input_overflow:
            print("PERINGATAN: Input audio overflow! Data audio mungkin hilang.", file=sys.stderr)
    indata_bytes = indata # Already bytes due to dtype='int16' in RawInputStream
    chunk_size_bytes = len(indata_bytes)
    if chunk_size_bytes == 0:
        return
    with buffer_lock:
        if write_pos + chunk_size_bytes > BUFFER_SIZE_BYTES:
            bytes_until_end = BUFFER_SIZE_BYTES - write_pos
            audio_ring_buffer[write_pos:BUFFER_SIZE_BYTES] = indata_bytes[:bytes_until_end]
            bytes_remaining = chunk_size_bytes - bytes_until_end
            audio_ring_buffer[0:bytes_remaining] = indata_bytes[bytes_until_end:]
            write_pos = bytes_remaining
        else:
            audio_ring_buffer[write_pos : write_pos + chunk_size_bytes] = indata_bytes
            write_pos += chunk_size_bytes
        total_written_bytes += chunk_size_bytes
        data_available.notify_all()

# --- Utility Functions ---

def connect_to_wifi(ssid, password):
    wifi = PyWiFi()
    iface = None
    try:
        ifaces = wifi.interfaces()
        if not ifaces:
            print("Tidak ada antarmuka Wi-Fi yang ditemukan.")
            return False
        iface = ifaces[0]
    except Exception as e:
        print(f"Error saat mengakses antarmuka Wi-Fi: {e}")
        return False

    iface.disconnect()
    time.sleep(1)

    try:
        iface.scan()
        print("Scanning Wi-Fi...")
        time.sleep(3)
        scan_results = iface.scan_results()
    except Exception as e:
        print(f"Error saat scan Wi-Fi: {e}")
        return False
    for network in scan_results:
        if network.ssid == ssid:
            print(f"SSID ditemukan: {ssid}")
            profile = Profile()
            profile.ssid = ssid
            profile.auth = const.AUTH_ALG_OPEN
            profile.akm.append(const.AKM_TYPE_WPA2PSK)
            profile.cipher = const.CIPHER_TYPE_CCMP
            profile.key = password
            iface.remove_all_network_profiles()
            temp_profile = iface.add_network_profile(profile)
            iface.connect(temp_profile)
            print(f"Mencoba terhubung ke {ssid}...")
            for _ in range(10):
                time.sleep(1)
                if iface.status() == const.IFACE_CONNECTED:
                    print("Berhasil terhubung ke Wi-Fi!")
                    return True
            print(f"Gagal terhubung ke {ssid} setelah beberapa percobaan.")
            return False
    print(f"SSID {ssid} tidak ditemukan.")
    return False

def is_internet_connected(timeout=2):
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=timeout)
        return True
    except OSError:
        return False
    except Exception:
        return False

def find_esp32_port():
    ports = serial.tools.list_ports.comports()
    print("Mencari port ESP32...")
    for port in ports:
        if "ACM" in port.device.upper() or "USB" in port.device.upper() or \
           (port.manufacturer and ("Silicon Labs" in port.manufacturer or "wch.cn" in port.manufacturer or "FTDI" in port.manufacturer or "Prolific" in port.manufacturer)) or \
           (port.description and ("USB-SERIAL CH340" in port.description.upper() or "CP210x" in port.description.upper() or "USB Serial Port" in port.description or "UART" in port.description.upper())):
            print(f"ESP32 kemungkinan ditemukan di {port.device} ({port.description})")
            return port.device
    print("ESP32 tidak ditemukan. Pastikan terhubung dan driver terinstal.")
    return None

def check_microphones():
    p = None
    try:
        p = pyaudio.PyAudio()
        print("Mikrofon yang terdeteksi:")
        found_mic = False
        default_mic_index = -1
        try:
            default_mic_info = p.get_default_input_device_info()
            default_mic_index = default_mic_info['index']
            print(f"  Default Input Device: ID {default_mic_index}, Nama: {default_mic_info['name']}")
        except IOError:
            print("  Tidak ada default input device yang terkonfigurasi di sistem.")

        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info.get("maxInputChannels", 0) > 0:
                print(f"  ID: {i}, Nama: {device_info['name']}{' (DEFAULT)' if i == default_mic_index else ''}")
                if i == default_mic_index or \
                   any(keyword in device_info["name"].lower() for keyword in ["microphone", "mic", "input", "capture", "respeaker", "realtek", "usb pnp sound device"]):
                    if not found_mic:
                         print(f"    -> Mikrofon yang sesuai/default ditemukan: {device_info['name']}")
                    found_mic = True
        if not found_mic:
            print("PERINGATAN: Tidak ada mikrofon yang dikenali (ReSpeaker, Realtek, default, dll.) terdeteksi. Pastikan mic terhubung dan dipilih sebagai default.")
        return found_mic
    except Exception as e:
        print(f"Error saat memeriksa mikrofon dengan PyAudio: {e}")
        return False
    finally:
        if p:
            p.terminate()

def check_speakers():
    p = None
    try:
        p = pyaudio.PyAudio()
        print("Speaker yang terdeteksi:")
        default_speaker_index = -1
        try:
            default_speaker_info = p.get_default_output_device_info()
            default_speaker_index = default_speaker_info['index']
            print(f"  Default Output Device: ID {default_speaker_index}, Nama: {default_speaker_info['name']}")
        except IOError:
            print("  Tidak ada default output device yang terkonfigurasi di sistem.")

        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info.get("maxOutputChannels", 0) > 0:
                print(f"  ID: {i}, Nama: {device_info['name']}{' (DEFAULT)' if i == default_speaker_index else ''}")
    except Exception as e:
        print(f"Error saat memeriksa speaker dengan PyAudio: {e}")
    finally:
        if p:
            p.terminate()

def record_audio_dynamic(duration_min=3, duration_max=6, silence_duration=1, sampling_rate=16000):
    chunk_duration = 0.5
    chunk_samples = int(chunk_duration * sampling_rate)
    total_audio = []
    min_samples = int(duration_min * sampling_rate)
    max_samples = int(duration_max * sampling_rate)
    silence_threshold = 0.01
    print("Mulai merekam (dinamis)...")
    try:
        with sd.InputStream(samplerate=sampling_rate, channels=1, dtype='float32', blocksize=chunk_samples) as stream:
            start_time = time.time()
            while len(total_audio) < min_samples:
                if time.time() - start_time > duration_max + 3:
                    print("Perekaman dinamis (min_samples) melebihi batas waktu maksimum.")
                    break
                try:
                    chunk, overflowed = stream.read(chunk_samples)
                    if overflowed:
                        print("PERINGATAN: Input audio overflow saat merekam perintah (min_samples loop).", file=sys.stderr)
                    total_audio.extend(chunk.flatten())
                except sd.CallbackStop:
                    print("Perekaman dinamis (min_samples) dihentikan oleh callback.")
                    break
                except Exception as e:
                     print(f"Error saat membaca stream (min_samples): {e}")
                     break

            last_audio_activity_time = time.time()
            while len(total_audio) < max_samples:
                 current_time = time.time()
                 if current_time - start_time > duration_max + 3:
                     print("Perekaman dinamis (max_samples) melebihi batas waktu maksimum.")
                     break
                 if current_time - last_audio_activity_time > silence_duration + chunk_duration:
                      print("Deteksi silence (timeout aktivitas), menghentikan perekaman dinamis.")
                      break
                 try:
                    chunk, overflowed = stream.read(chunk_samples)
                    if overflowed:
                        print("PERINGATAN: Input audio overflow saat merekam perintah (max_samples loop).", file=sys.stderr)
                    total_audio.extend(chunk.flatten())
                    if np.mean(np.abs(chunk)) >= silence_threshold:
                        last_audio_activity_time = current_time
                 except sd.CallbackStop:
                    print("Perekaman dinamis (max_samples) dihentikan oleh callback.")
                    break
                 except Exception as e:
                     print(f"Error saat membaca stream (max_samples): {e}")
                     break
        print("Selesai merekam (dinamis).")
        return np.array(total_audio, dtype=np.float32)
    except sd.PortAudioError as e:
        print(f"Error SoundDevice saat merekam (dinamis): {e}")
        return np.array([], dtype=np.float32)
    except Exception as e:
        print(f"Error tak terduga saat merekam (dinamis): {e}")
        return np.array([], dtype=np.float32)

def online_stt_recognize(audio_data_sr, language, result_container):
    try:
        recognizer = sr.Recognizer()
        result_container['text'] = recognizer.recognize_google(audio_data_sr, language=language)
    except sr.UnknownValueError:
        result_container['text'] = ""
    except sr.RequestError as e:
        print(f"Google STT: Error koneksi/API; {e}")
        result_container['text'] = ""
    except Exception as e:
        print(f"Google STT: Error tak terduga: {e}")
        result_container['text'] = ""

def record_audio(duration=3, sampling_rate=16000, noise_reduce=True, dynamic=True):
    if dynamic:
        audio_data = record_audio_dynamic(duration_min=duration, duration_max=6, silence_duration=1, sampling_rate=sampling_rate)
    else:
        print(f"Mulai merekam (durasi tetap: {duration} detik)...")
        try:
            num_samples = int(duration * sampling_rate)
            recording = sd.rec(
                num_samples,
                samplerate=sampling_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            audio_data = recording.flatten()
            print("Selesai merekam (durasi tetap).")
        except sd.PortAudioError as e:
            print(f"Error SoundDevice saat merekam (fixed): {e}")
            return np.array([], dtype=np.float32)
        except Exception as e:
            print(f"Error tak terduga saat merekam (fixed): {e}")
            return np.array([], dtype=np.float32)

    if audio_data.size == 0:
        return np.array([], dtype=np.float32)

    if noise_reduce and audio_data.size > int(0.5 * sampling_rate):
        noise_sample_duration = min(0.5, audio_data.size / sampling_rate * 0.2)
        noise_sample_samples = int(noise_sample_duration * sampling_rate)
        if noise_sample_samples > 0:
             noise_sample = audio_data[:noise_sample_samples]
             audio_data = reduce_noise(audio_data, noise_sample, sampling_rate)
        else:
            print("Audio terlalu pendek untuk mengumpulkan sampel noise.")
    elif noise_reduce:
        print("Audio terlalu pendek untuk noise reduction, dilewati.")
    return audio_data

def reduce_noise(audio, noise_sample, sr_rate, prop_decrease=0.8):
    try:
        if noise_sample.size == 0 or audio.size <= noise_sample.size:
            return audio
        audio_float = audio.astype(np.float32)
        noise_sample_float = noise_sample.astype(np.float32)

        return nr.reduce_noise(
            y=audio_float,
            y_noise=noise_sample_float,
            sr=sr_rate,
            prop_decrease=prop_decrease,
            stationary=True
        )
    except Exception as e:
        print(f"Error dalam noise reduction: {e}")
        return audio

def save_audio(audio_array, filename, sampling_rate=16000):
    try:
        if audio_array.size == 0:
            print(f"Tidak ada data audio untuk disimpan ke {filename}.")
            return
        if audio_array.dtype != np.float32:
            audio_array_float32 = audio_array.astype(np.float32)
            if np.issubdtype(audio_array.dtype, np.integer):
                 audio_array_float32 = audio_array_float32 / np.iinfo(audio_array.dtype).max
        else:
            audio_array_float32 = audio_array

        max_abs_val = np.max(np.abs(audio_array_float32))
        if max_abs_val == 0:
             audio_normalized_int16 = np.zeros_like(audio_array_float32, dtype=np.int16)
        else:
            audio_normalized_int16 = np.int16(audio_array_float32 * 32767 / max_abs_val)
        wav.write(filename, sampling_rate, audio_normalized_int16)
        print(f"Audio disimpan ke {filename}")
    except Exception as e:
        print(f"Gagal menyimpan audio ke {filename}: {e}")

def is_audio_present(audio_array_float32, threshold=0.005):
    if audio_array_float32.size == 0:
        print("Audio array kosong, tidak ada audio terdeteksi.")
        return False
    rms = np.sqrt(np.mean(audio_array_float32**2))
    print(f"Audio presence check RMS value: {rms:.4f} (Threshold: {threshold})")
    return rms > threshold

def transcribe_audio(audio_array_float32, sampling_rate=16000):
    if audio_array_float32.size == 0:
        print("Tidak ada data audio untuk ditranskripsi (offline).")
        return {'text': '', 'processing_time': 0}
    try:
        if audio_array_float32.dtype != np.float32:
            audio_array_float32 = audio_array_float32.astype(np.float32)

        inputs = stt_processor(
            audio_array_float32,
            sampling_rate=sampling_rate,
            return_tensors="np"
        )
        features = ctranslate2.StorageView.from_array(inputs.input_features)
        prompt_tokens = [
            "<|startoftranscript|>",
            "<|transcribe|>",
            "<|notimestamps|>",
        ]
        prompt = stt_processor.tokenizer.convert_tokens_to_ids(prompt_tokens)
        start_time = time.time()
        results = stt_model.generate(features, [prompt])
        processing_time = time.time() - start_time
        cleaned_transcription = stt_processor.decode(
            results[0].sequences_ids[0],
            skip_special_tokens=True
        )
        return {
            'text': cleaned_transcription.strip(),
            'processing_time': processing_time
        }
    except Exception as e:
        print(f"Error saat transkripsi audio offline: {e}")
        return {'text': '', 'processing_time': 0}

def predict_intent(is_online, text):
    def predict_online(text_input):
        if not GEMINI_URL or not GEMINI_API_KEY:
            print("Gemini API URL atau Key tidak dikonfigurasi. Beralih ke model offline.")
            return predict_offline(text_input)
        prompt = (
            f"Task: Intent Classification.\n"
            f"Input text: \"{text_input}\"\n"
            f"Available intents: {json.dumps(nlp_label_list)}\n"
            f"Output only the single most appropriate intent from the list. "
            f"If the input is ambiguous, irrelevant to the intents, or implies multiple intents, output \"tidak relevan\". "
            f"Analyze carefully, considering implied meanings."
        )
        data = {"contents" : [{"parts" : [{"text" : prompt}]}]}
        try:
            response = requests.post(GEMINI_URL, json=data, timeout=7)
            response.raise_for_status()
            response_json = response.json()
            if "candidates" in response_json and \
               len(response_json["candidates"]) > 0 and \
               "content" in response_json["candidates"][0] and \
               "parts" in response_json["candidates"][0]["content"] and \
               len(response_json["candidates"][0]["content"]["parts"]) > 0 and \
               "text" in response_json["candidates"][0]["content"]["parts"][0]:
                predicted_intent = response_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                if predicted_intent in nlp_label_list:
                    return predicted_intent
                else:
                    for label in nlp_label_list:
                        if label in predicted_intent:
                            print(f"Gemini returned '{predicted_intent}', matched to '{label}' via substring.")
                            return label
                    print(f"Gemini returned '{predicted_intent}', not in nlp_label_list and no substring match. Defaulting to 'tidak relevan'.")
                    return "tidak relevan"
            else:
                print("Gemini API: Respons tidak memiliki struktur yang diharapkan.")
                return "tidak relevan"
        except requests.exceptions.Timeout:
            print("Gemini API: Timeout.")
            return "tidak relevan"
        except requests.exceptions.HTTPError as e:
            print(f"Gemini API: HTTP Error: {e}")
            return "tidak relevan"
        except requests.exceptions.RequestException as e:
            print(f"Gemini API: Request Exception: {e}")
            return "tidak relevan"
        except json.JSONDecodeError:
            print("Gemini API: Gagal parse JSON response.")
            return "tidak relevan"
        except KeyError as e:
            print(f"Gemini API: KeyError saat parsing response - {e}.")
            return "tidak relevan"
        except Exception as e:
            print(f"Gemini API: Terjadi error tak terduga: {e}")
            return "tidak relevan"

    def predict_offline(text_input):
        try:
            inputs = nlp_tokenizer(
                text_input,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to("cpu")
            with torch.no_grad():
                outputs = nlp_model(**inputs)
            logits = outputs.logits
            pred_id = torch.argmax(logits, dim=-1).item()
            return id_to_label[pred_id]
        except Exception as e:
            print(f"Error saat prediksi intent offline: {e}")
            return "tidak relevan"
    if is_online:
        return predict_online(text)
    else:
        return predict_offline(text)

def send_to_esp(command):
    global ESP
    try:
        if not USING_ESP:
            return True
        elif ESP is not None and ESP.is_open:
            ESP.write((command + "\n").encode('utf-8'))
            return True
        else:
            print("Port serial ESP tidak terbuka atau ESP belum diinisialisasi.")
            if USING_ESP and (ESP is None or not ESP.is_open):
                print("Mencoba menghubungkan kembali ke ESP32...")
                esp_port = find_esp32_port()
                if esp_port:
                    try:
                        ESP = serial.Serial(esp_port, 115200, timeout=1)
                        print(f"Berhasil terhubung kembali ke ESP32 di {esp_port}.")
                        ESP.write((command + "\n").encode('utf-8'))
                        return True
                    except serial.SerialException as se:
                        print(f"Gagal re-koneksi ke ESP32: {se}")
                        ESP = None
                        return False
                else:
                    print("Tidak dapat menemukan port ESP32 untuk re-koneksi.")
                    return False
            return False
    except serial.SerialException as e:
        print(f"Gagal mengirim ke ESP: {e}. Port mungkin terputus.")
        ESP = None
        return False
    except Exception as e:
        print(f"Error tak terduga saat mengirim ke ESP: {e}")
        return False

def intent_feedback(intent, predicted_language="Indonesian"):
    global program_state, main_loop_active_flag

    if intent == program_state.get("last_command_for_feedback_purpose") and intent not in ["wake", "off", "system_on"]:
        print(f"Intent '{intent}' sama dengan feedback terakhir. Mengabaikan duplikasi feedback audio.")
        return
    
    feedback_audio_base_path = os.path.join(this_file_dir, "feedback_audio")
    colors = ["merah", "hijau", "biru", "lavender", "magenta", "pink", "violet", "aqua", "kuning", "emas", "abu"]
    audio_file_to_play = None
    executed_successfully = True

    lang_path = os.path.join(feedback_audio_base_path, predicted_language)
    if not os.path.isdir(lang_path):
        print(f"Peringatan: Direktori feedback audio '{predicted_language}' tidak ditemukan. Fallback ke 'Indonesian'.")
        lang_path = os.path.join(feedback_audio_base_path, "Indonesian")
        if not os.path.isdir(lang_path):
            print(f"Peringatan: Direktori feedback audio default 'Indonesian' juga tidak ditemukan. Tidak ada feedback audio.")
            return

    suffix = "_pria" if program_state["gender"] == "pria" else "_wanita"

    if "nyalakan lampu" in intent:
        real_intent_color = None
        for color in colors:
            if color in intent.lower():
                real_intent_color = color
                break
        if real_intent_color:
            if send_to_esp(real_intent_color.upper()):
                audio_file_to_play = os.path.join(lang_path, f"ganti_warna{suffix}.mp3")
                program_state["current_light_state"] = real_intent_color.upper()
                program_state["last_command"] = intent
            else: executed_successfully = False
        elif intent == "nyalakan lampu":
            if send_to_esp("ON"):
                audio_file_to_play = os.path.join(lang_path, f"menyalakan_lampu{suffix}.mp3")
                program_state["current_light_state"] = "ON"
                program_state["last_command"] = intent
            else: executed_successfully = False
    elif "matikan lampu" in intent:
        if send_to_esp("OFF"):
            audio_file_to_play = os.path.join(lang_path, f"mematikan_lampu{suffix}.mp3")
            program_state["current_light_state"] = "OFF"
            program_state["last_command"] = intent
        else: executed_successfully = False
    elif "nyalakan mode senang" in intent:
        if send_to_esp("HAPPY"):
            audio_file_to_play = os.path.join(lang_path, f"senang{suffix}.mp3")
            program_state["last_command"] = intent
        else: executed_successfully = False
    elif "nyalakan mode sad" in intent:
        if send_to_esp("SAD"):
            audio_file_to_play = os.path.join(lang_path, f"sedih{suffix}.mp3")
            program_state["last_command"] = intent
        else: executed_successfully = False
    elif "gender ke wanita" in intent:
        program_state["gender"] = "wanita"
        audio_file_to_play = os.path.join(feedback_audio_base_path, "Indonesian", "ganti_suara_ke_wanita.mp3")
        print("Gender diubah ke wanita.")
        program_state["last_command"] = intent
    elif "gender ke pria" in intent:
        program_state["gender"] = "pria"
        audio_file_to_play = os.path.join(feedback_audio_base_path, "Indonesian", "ganti_suara_ke_pria.mp3")
        print("Gender diubah ke pria.")
        program_state["last_command"] = intent
    elif "wake" in intent:
        audio_file_to_play = os.path.join(feedback_audio_base_path, "ping_berbicara.mp3")
    elif "off" in intent:
        audio_file_to_play = os.path.join(feedback_audio_base_path, "off_to_wakeword.mp3")
    elif "system_on" in intent:
        audio_file_to_play = os.path.join(feedback_audio_base_path, "system_on.mp3")
    elif "tidak relevan" in intent:
        if program_state.get("last_command_for_feedback_purpose") != "tidak relevan":
            audio_file_to_play = os.path.join(lang_path, f"perintah_tidak_dikenali{suffix}.mp3")
        print("Perintah tidak dikenali atau tidak ada suara signifikan.")
    else:
        print(f"Intent '{intent}' tidak memiliki feedback audio spesifik yang terkonfigurasi.")


    if audio_file_to_play and executed_successfully:
        if os.path.exists(audio_file_to_play):
            try:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                    time.sleep(0.05)

                pygame.mixer.music.load(audio_file_to_play)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy() and main_loop_active_flag.is_set():
                    time.sleep(0.05)
                if not main_loop_active_flag.is_set() and pygame.mixer.music.get_busy():
                     pygame.mixer.music.stop()
                program_state["last_command_for_feedback_purpose"] = intent
            except pygame.error as e:
                print(f"Gagal memainkan file audio '{audio_file_to_play}': {e}")
            except Exception as e:
                print(f"Error tak terduga saat memainkan audio: {e}")
        else:
            print(f"File audio feedback tidak ditemukan: {audio_file_to_play}")
    elif not executed_successfully:
        print(f"Gagal mengirim perintah '{intent}' ke ESP. Feedback audio tidak dimainkan.")

def initialize_system():
    global ESP, wake_model
    print("Inisialisasi sistem...")
    if USING_ESP:
        esp_port = find_esp32_port()
        if esp_port:
            try:
                ESP = serial.Serial(esp_port, 115200, timeout=1)
                print(f"Berhasil terhubung ke ESP32 di {esp_port}")
                time.sleep(1.5)
            except serial.SerialException as e:
                print(f"Gagal terhubung ke ESP32: {e}. Mode tanpa ESP akan digunakan jika memungkinkan.")
                ESP = None
            except Exception as e:
                print(f"Error tak terduga saat inisialisasi ESP32: {e}")
                ESP = None
        else:
            print("ESP32 tidak terdeteksi. Mode tanpa ESP akan digunakan jika memungkinkan.")
            ESP = None
    else:
        print("Mode tanpa ESP32 diaktifkan.")
        ESP = None

    try:
        wake_model_path_name = "vosk-model-en-us-0.22-lgraph"
        wake_model = Model(model_name=wake_model_path_name)
        print("Model Vosk (wake word) berhasil dimuat.")
    except Exception as e:
        print(f"FATAL: Gagal memuat model Vosk wake word: {e}. Pastikan model ada di path yang benar atau nama model standar.")
        sys.exit(1)

    if not check_microphones():
        print("PERINGATAN PENTING: Mikrofon yang sesuai tidak terdeteksi atau tidak ada default. Program mungkin tidak dapat menerima input suara.")
    check_speakers()
    intent_feedback("system_on")
    print("Sistem siap.")
    return True

ONLINE_WAKE_WORD_CHECK_INTERVAL = 2
ONLINE_WAKE_WORD_RECORD_DURATION = 4
SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2

def run_vosk_wake_word_detector(wake_event, vosk_recognizer, main_loop_flag, samplerate, result_language_container):
    global buffer_lock, data_available, total_written_bytes, audio_ring_buffer, BUFFER_SIZE_BYTES
    print("Vosk wake word detector thread dimulai.")
    try:
        with buffer_lock:
             if total_written_bytes == 0:
                  print("Vosk thread menunggu data audio pertama...")
                  data_available.wait(timeout=5.0)
             current_vosk_read_pos = total_written_bytes

        while not wake_event.is_set() and main_loop_flag.is_set():
            bytes_to_process = 0
            chunk_data = None
            with buffer_lock:
                bytes_available_total = total_written_bytes - current_vosk_read_pos
                if bytes_available_total == 0:
                    data_available.wait(timeout=0.1)
                    bytes_available_total = total_written_bytes - current_vosk_read_pos
                if bytes_available_total > 0:
                    bytes_to_process = bytes_available_total
                    read_start_pos_total = current_vosk_read_pos
                    start_index_in_buffer = read_start_pos_total % BUFFER_SIZE_BYTES
                    if start_index_in_buffer + bytes_to_process <= BUFFER_SIZE_BYTES:
                        chunk_data = audio_ring_buffer[start_index_in_buffer : start_index_in_buffer + bytes_to_process]
                    else:
                        bytes_until_end = BUFFER_SIZE_BYTES - start_index_in_buffer
                        chunk1 = audio_ring_buffer[start_index_in_buffer : BUFFER_SIZE_BYTES]
                        bytes_from_start = bytes_to_process - bytes_until_end
                        chunk2 = audio_ring_buffer[0 : bytes_from_start]
                        chunk_data = chunk1 + chunk2
                    if len(chunk_data) != bytes_to_process:
                        print(f"PERINGATAN VOSK: Mismatch in chunk_data length. Expected {bytes_to_process}, got {len(chunk_data)}")
                        chunk_data = None
                    if chunk_data:
                        current_vosk_read_pos += bytes_to_process
            if wake_event.is_set() or not main_loop_flag.is_set(): break
            if chunk_data and len(chunk_data) > 0:
                if vosk_recognizer.AcceptWaveform(bytes(chunk_data)):
                    result_json = vosk_recognizer.Result()
                    result = json.loads(result_json)
                    if 'text' in result and result['text']:
                        detected_text = result['text'].lower().strip()
                        lang = None
                        if "oke toyota" in detected_text or "okay toyota" in detected_text:
                            lang = "Indonesian"
                        elif "hello toyota" in detected_text:
                            lang = "English"
                        elif ("hai toyota" in detected_text) or \
                             ("moshi toyota" in detected_text) or \
                             ("moshi moshi toyota" in detected_text):
                            lang = "Japanese"
                        if lang:
                            print(f"Vosk mendeteksi wake word! Bahasa: {lang} (Teks: '{detected_text}')")
                            if not wake_event.is_set():
                                result_language_container['language'] = lang
                                wake_event.set()
                            break
            if bytes_to_process == 0 and not (wake_event.is_set() or not main_loop_flag.is_set()):
                 time.sleep(0.01)
    except Exception as e:
        if main_loop_active_flag.is_set():
            print(f"Error di Vosk wake word detector thread: {e}")
            import traceback
            traceback.print_exc()
    finally:
        print("Vosk wake word detector thread berhenti.")

def run_online_wake_word_check(wake_event, samplerate, result_language_container, main_loop_flag):
    global total_written_bytes, buffer_lock, audio_ring_buffer, BUFFER_SIZE_BYTES, SAMPLE_WIDTH_BYTES, write_pos
    if wake_event.is_set() or not main_loop_flag.is_set():
        return
    print("Memulai pengecekan wake word online (1x STT id-ID)...")
    bytes_to_collect_requested = int(ONLINE_WAKE_WORD_RECORD_DURATION * samplerate * SAMPLE_WIDTH_BYTES)
    collected_bytes_data = bytearray()
    with buffer_lock:
        bytes_available_in_buffer = min(BUFFER_SIZE_BYTES, total_written_bytes)
        bytes_to_collect = min(bytes_to_collect_requested, bytes_available_in_buffer)
        if bytes_to_collect == 0:
             print("Online check: Tidak ada data audio di buffer untuk dicek (0 bytes available).")
             return
        current_write_pos_snapshot = write_pos
        if current_write_pos_snapshot >= bytes_to_collect:
            start_read_idx = current_write_pos_snapshot - bytes_to_collect
            collected_bytes_data.extend(audio_ring_buffer[start_read_idx : current_write_pos_snapshot])
        else:
            bytes_from_end = bytes_to_collect - current_write_pos_snapshot
            collected_bytes_data.extend(audio_ring_buffer[BUFFER_SIZE_BYTES - bytes_from_end : BUFFER_SIZE_BYTES])
            collected_bytes_data.extend(audio_ring_buffer[0 : current_write_pos_snapshot])
        if len(collected_bytes_data) != bytes_to_collect:
             print(f"PERINGATAN KRITIS: Online check collected unexpected number of bytes. Expected {bytes_to_collect}, got {len(collected_bytes_data)}. Discarding data.")
             return
    if wake_event.is_set() or not main_loop_flag.is_set():
        print("Online check: Pengecekan dibatalkan setelah mengumpulkan data.")
        return
    audio_int16_np = np.frombuffer(bytes(collected_bytes_data), dtype=np.int16)
    audio_float32_np = audio_int16_np.astype(np.float32) / 32768.0
    print(f"Online check: Audio terkumpul {len(collected_bytes_data)} bytes ({len(collected_bytes_data)/(samplerate*SAMPLE_WIDTH_BYTES):.2f}s). Memanggil is_audio_present...")
    if not is_audio_present(audio_float32_np, threshold=0.004):
        print("Online check: Audio tidak signifikan terdeteksi (setelah is_audio_present).")
        return
    try:
        audio_data_sr = sr.AudioData(
            bytes(collected_bytes_data),
            sample_rate=samplerate,
            sample_width=SAMPLE_WIDTH_BYTES
        )
        stt_result_container = {}
        stt_thread = threading.Thread(target=online_stt_recognize, args=(audio_data_sr, "id-ID", stt_result_container))
        stt_thread.daemon = True
        stt_thread.start()
        stt_thread.join(timeout=4.5)
        if stt_thread.is_alive():
            print("Online STT (wake word) timeout.")
            return
        if wake_event.is_set() or not main_loop_flag.is_set(): return
        text_transcribed = stt_result_container.get('text', "").lower().strip()
        if text_transcribed:
            print(f"Online check STT (id-ID): '{text_transcribed}'")
            lang_detected_online = None
            if "oke toyota" in text_transcribed or "okay toyota" in text_transcribed:
                lang_detected_online = "Indonesian"
            elif "hello toyota" in text_transcribed:
                lang_detected_online = "English"
            elif (("hai toyota" in text_transcribed)) or \
                 ("moshi toyota" in text_transcribed) or \
                 ("moshi moshi toyota" in text_transcribed) or \
                 ("mosi toyota" in text_transcribed):
                lang_detected_online = "Japanese"
            if lang_detected_online:
                print(f"Online STT (id-ID) mendeteksi wake word! Bahasa ditentukan: {lang_detected_online}")
                if not wake_event.is_set():
                    result_language_container['language'] = lang_detected_online
                    wake_event.set()
    except Exception as e:
        if main_loop_active_flag.is_set():
            print(f"Error pada saat pengecekan wake word online: {e}")
            import traceback
            traceback.print_exc()
    finally:
        pass

# --- Global variable for command mode timer and its lock ---
command_mode_start_time_shared = 0.0
command_mode_timer_lock = threading.Lock()

# --- MAIN PROGRAM ---
if __name__ == "__main__":
    main_loop_active_flag = threading.Event()
    main_loop_active_flag.set()

    if not initialize_system():
        print("FATAL: Gagal melakukan inisialisasi sistem. Program berhenti.")
        sys.exit(1)

    try:
        device_info = sd.query_devices(None, "input")
        SAMPLERATE = int(device_info.get("default_samplerate", 16000))
        print(f"Menggunakan samplerate default input device: {SAMPLERATE} Hz")
    except Exception as e:
        print(f"PERINGATAN: Gagal mendapatkan default samplerate: {e}. Menggunakan default {SAMPLE_RATE} Hz.")
        SAMPLERATE = SAMPLE_RATE

    audio_stream = None
    vosk_thread = None
    online_check_thread = None
    # active_command_threads list is removed as command processing is now sequential

    try:
        while main_loop_active_flag.is_set():
            print(f"\nMenunggu wake word (Offline Vosk kontinu, Online Google STT periodik [id-ID setiap {ONLINE_WAKE_WORD_CHECK_INTERVAL}s])...")
            wake_word_detected_event = threading.Event()
            detected_language_container = {'language': "Indonesian"}

            with buffer_lock:
                 audio_ring_buffer = bytearray(BUFFER_SIZE_BYTES)
                 write_pos = 0
                 total_written_bytes = 0

            blocksize_callback_samples = int(SAMPLERATE * 0.2)
            blocksize_callback_bytes = blocksize_callback_samples * SAMPLE_WIDTH_BYTES
            if BUFFER_SIZE_BYTES < blocksize_callback_bytes * 5:
                 print(f"PERINGATAN: BUFFER_SIZE_BYTES ({BUFFER_SIZE_BYTES}) mungkin terlalu kecil untuk blocksize callback ({blocksize_callback_bytes}). Pertimbangkan untuk menambah BUFFER_SIZE_BYTES.")

            try:
                audio_stream = sd.RawInputStream(
                    samplerate=SAMPLERATE,
                    blocksize=blocksize_callback_samples,
                    device=None,
                    dtype="int16",
                    channels=1,
                    callback=callback
                )
                audio_stream.start()
                print("Stream audio dimulai untuk deteksi wake word.")
            except sd.PortAudioError as pae:
                if not main_loop_active_flag.is_set(): break
                print(f"SoundDevice Error saat memulai stream: {pae}. Mencoba lagi dalam 2 detik...")
                time.sleep(2)
                continue
            except Exception as e_stream_start:
                 if not main_loop_active_flag.is_set(): break
                 print(f"Error tak terduga saat memulai stream audio: {e_stream_start}. Mencoba lagi dalam 2 detik...")
                 time.sleep(2)
                 continue

            vosk_grammar = ["oke toyota", "hello toyota", "hai toyota", "moshi toyota", "moshi moshi toyota", "[unk]"]
            vosk_recognizer = KaldiRecognizer(wake_model, SAMPLERATE, json.dumps(vosk_grammar, ensure_ascii=False))

            vosk_thread = threading.Thread(
                target=run_vosk_wake_word_detector,
                args=(wake_word_detected_event, vosk_recognizer, main_loop_active_flag, SAMPLERATE, detected_language_container)
            )
            vosk_thread.daemon = True
            vosk_thread.start()

            last_online_check_time = time.time()
            while not wake_word_detected_event.is_set() and main_loop_active_flag.is_set():
                current_time = time.time()
                if is_internet_connected(timeout=0.5) and \
                   (current_time - last_online_check_time > ONLINE_WAKE_WORD_CHECK_INTERVAL):
                    if online_check_thread is None or not online_check_thread.is_alive():
                        last_online_check_time = current_time
                        online_check_thread = threading.Thread(
                            target=run_online_wake_word_check,
                            args=(wake_word_detected_event, SAMPLERATE, detected_language_container, main_loop_active_flag)
                        )
                        online_check_thread.daemon = True
                        online_check_thread.start()
                time.sleep(0.05)

            print("Keluar dari loop deteksi wake word (event set or shutdown).")
            if audio_stream and audio_stream.active:
                 print("Menghentikan stream audio wake word...")
                 audio_stream.stop()
            if audio_stream and not audio_stream.closed:
                 audio_stream.close()
                 audio_stream = None
                 print("Stream audio wake word ditutup.")

            wake_word_detected_event.set()
            if vosk_thread and vosk_thread.is_alive():
                 print("Menunggu Vosk thread selesai...")
                 vosk_thread.join(timeout=2.0)
                 if vosk_thread.is_alive():
                      print("PERINGATAN: Vosk thread tidak berhenti tepat waktu setelah stop stream.")
            if online_check_thread and online_check_thread.is_alive():
                 print("Menunggu online check thread selesai...")
                 online_check_thread.join(timeout=ONLINE_WAKE_WORD_RECORD_DURATION + 5.0)
                 if online_check_thread.is_alive():
                      print("PERINGATAN: Online check thread tidak berhenti tepat waktu.")

            if wake_word_detected_event.is_set() and detected_language_container.get('language') and main_loop_active_flag.is_set():
                program_state["predicted_language_from_wake_word"] = detected_language_container['language']
                print(f"Wake word terdeteksi! Bahasa yang digunakan: {program_state['predicted_language_from_wake_word']}")
                intent_feedback("wake", program_state["predicted_language_from_wake_word"])

                with command_mode_timer_lock:
                    command_mode_start_time_shared = time.time()
                command_mode_timeout = 11
                current_predicted_language = program_state["predicted_language_from_wake_word"]
                
                # Command loop (now sequential)
                while main_loop_active_flag.is_set():
                    current_loop_time_cmd = 0.0
                    with command_mode_timer_lock:
                        current_loop_time_cmd = time.time()
                        if current_loop_time_cmd - command_mode_start_time_shared >= command_mode_timeout:
                            print("Waktu mode perintah habis.")
                            break 

                    remaining_time = command_mode_timeout - (current_loop_time_cmd - command_mode_start_time_shared)
                    print(f"\nSilahkan ucapkan perintah (Bahasa: {current_predicted_language}, Sisa waktu: {remaining_time:.0f} detik)...")
                    
                    command_sampling_rate = 16000
                    recorded_audio_cmd_float32 = record_audio(
                        duration=3,
                        sampling_rate=command_sampling_rate,
                        dynamic=True,
                        noise_reduce=True
                    )

                    if not main_loop_active_flag.is_set(): break

                    if is_audio_present(recorded_audio_cmd_float32, threshold=0.003):
                        # --- Inlined command processing logic ---
                        internet_conn = is_internet_connected(timeout=1.0)
                        text_transcribed = ""
                        stt_processing_time = 0
                        intent_prediction_mode = internet_conn

                        if internet_conn:
                            try:
                                stt_online_start_time = time.time()
                                audio_int16_cmd = (np.clip(recorded_audio_cmd_float32, -1.0, 1.0) * 32767).astype(np.int16)
                                audio_data_sr_cmd = sr.AudioData(
                                    audio_int16_cmd.tobytes(),
                                    sample_rate=command_sampling_rate,
                                    sample_width=SAMPLE_WIDTH_BYTES
                                )
                                lang_code_google_cmd = {"Indonesian": "id-ID", "English": "en-US", "Japanese": "ja-JP"}.get(current_predicted_language, "id-ID")
                                
                                result_container_cmd = {}
                                # STT is now blocking in the main thread
                                online_stt_recognize(audio_data_sr_cmd, lang_code_google_cmd, result_container_cmd)
                                # No thread join needed, it's synchronous

                                if not main_loop_active_flag.is_set(): break

                                text_transcribed = result_container_cmd.get('text', "").strip()
                                stt_processing_time = time.time() - stt_online_start_time
                                if not text_transcribed:
                                    print("STT Online (perintah) tidak menghasilkan teks.")
                                    # Fallback will be handled by the empty text_transcribed check later
                                else:
                                    print(f"STT Online (perintah) berhasil: '{text_transcribed}'")
                            
                            except Exception as e_stt_online_cmd:
                                if not main_loop_active_flag.is_set(): break
                                print(f"Gagal STT online (perintah): {e_stt_online_cmd}. Menggunakan STT offline...")
                                offline_result_cmd = transcribe_audio(recorded_audio_cmd_float32, sampling_rate=command_sampling_rate)
                                text_transcribed = offline_result_cmd["text"]
                                stt_processing_time = offline_result_cmd["processing_time"]
                                intent_prediction_mode = False
                        else: 
                            print("Tidak ada koneksi internet. Menggunakan STT offline (Whisper CT2) untuk perintah...")
                            offline_result_cmd = transcribe_audio(recorded_audio_cmd_float32, sampling_rate=command_sampling_rate)
                            text_transcribed = offline_result_cmd["text"]
                            stt_processing_time = offline_result_cmd["processing_time"]

                        if not main_loop_active_flag.is_set(): break

                        if not text_transcribed.strip():
                            print("Tidak ada teks perintah yang berhasil ditranskripsi.")
                            intent_feedback("tidak relevan", current_predicted_language)
                        else:
                            print(f"Transkripsi perintah: '{text_transcribed}' (Waktu STT: {stt_processing_time:.2f} dtk, Mode Intent: {'Online' if intent_prediction_mode else 'Offline'})")
                            predicted_intent = predict_intent(intent_prediction_mode, text_transcribed)
                            print(f"Prediksi intent: '{predicted_intent}'")
                            
                            # program_state["last_command"] is updated by intent_feedback if successful
                            intent_feedback(predicted_intent, current_predicted_language)

                            # Timer reset logic (from original process_command_audio_in_thread)
                            # Use program_state["last_command"] which is updated by intent_feedback
                            # or more directly, use the predicted_intent for this check.
                            # The original logic was:
                            # if predicted_intent != "tidak relevan" or program_state["last_command"] != predicted_intent:
                            # Let's use a slightly more robust check based on whether the feedback was for "tidak relevan"
                            # or if the new intent is different from the one that just got feedback.
                            if predicted_intent != "tidak relevan" or program_state.get("last_command_for_feedback_purpose") != predicted_intent:
                                with command_mode_timer_lock:
                                    print(f"Perintah '{predicted_intent}' dianggap relevan atau berbeda, mereset timer mode perintah.")
                                    command_mode_start_time_shared = time.time()
                            else:
                                print(f"Perintah '{predicted_intent}' tidak mereset timer.")
                        # --- End of inlined command processing logic ---
                    else:
                        print("Tidak ada suara signifikan terdeteksi untuk perintah.")
                    
                    if not main_loop_active_flag.is_set(): break
                    time.sleep(0.1) 


                if main_loop_active_flag.is_set():
                    print("Kembali ke mode deteksi wake word.")
                    intent_feedback("off", current_predicted_language)

            elif not main_loop_active_flag.is_set():
                print("Program dihentikan saat menunggu wake word atau setelahnya.")
                break 
            else: 
                print("Tidak ada wake word terdeteksi atau bahasa tidak diset. Memulai ulang siklus.")
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nCtrl+C terdeteksi. Menghentikan program...")
    except Exception as e_main:
         print(f"\nError tak terduga di main loop: {e_main}")
         import traceback
         traceback.print_exc()
    finally:
        print("Membersihkan resource sebelum keluar...")
        main_loop_active_flag.clear()

        if audio_stream and audio_stream.active:
            print("Menghentikan stream audio final (jika masih aktif)...")
            audio_stream.stop()
        if audio_stream and not audio_stream.closed:
             audio_stream.close()
             print("Stream audio final ditutup.")

        if 'vosk_thread' in locals() and vosk_thread and vosk_thread.is_alive():
            print("Menunggu Vosk thread (final)...")
            vosk_thread.join(timeout=2.0)
        if 'online_check_thread' in locals() and online_check_thread and online_check_thread.is_alive():
            print("Menunggu online check thread (final)...")
            online_check_thread.join(timeout=ONLINE_WAKE_WORD_RECORD_DURATION + 5.0)
        
        # No active_command_threads to join anymore

        if ESP and ESP.is_open:
            ESP.close()
            print("Port serial ESP ditutup.")
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            print("Pygame mixer dihentikan.")
        sd.stop()
        print("Program selesai.")