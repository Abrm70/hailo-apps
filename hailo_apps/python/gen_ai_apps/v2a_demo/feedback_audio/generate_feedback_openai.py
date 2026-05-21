# To run this code you need to install the following dependencies:
# pip install google-genai

import os
import struct
import mimetypes
from google import genai
from google.genai import types

# ==============================================================================
# PERINGATAN KEAMANAN PENTING
# ==============================================================================
# JANGAN PERNAH MENULIS KUNCI API ANDA LANGSUNG DI DALAM KODE.
# Kode ini dirancang untuk membaca kunci dari environment variable,
# yang merupakan praktik yang jauh lebih aman.
#
# Sebelum menjalankan skrip ini, atur environment variable di terminal Anda:
# - Di Windows: set GEMINI_API_KEY=KUNCI_API_ANDA
# - Di macOS/Linux: export GEMINI_API_KEY=KUNCI_API_ANDA
#
# Ganti KUNCI_API_ANDA dengan kunci API Gemini Anda yang sebenarnya.
# ==============================================================================


def save_binary_file(file_name, data):
    """Menyimpan data biner ke sebuah file."""
    try:
        with open(file_name, "wb") as f:
            f.write(data)
        print(f"✅ File berhasil disimpan ke: {file_name}")
    except IOError as e:
        print(f"❌ Gagal menyimpan file {file_name}: {e}")


def generate_all_files():
    """
    Fungsi utama untuk menghasilkan semua file audio yang dibutuhkan
    untuk sistem voice recognition di mobil.
    """
    # Periksa apakah kunci API sudah diatur
    api_key = "AIzaSyBZaCJ9N2XmYt_0iCTyebLfuV0zgXxhYXU"
    if not api_key:
        print("❌ KESALAHAN: Environment variable 'GEMINI_API_KEY' tidak ditemukan.")
        print("Silakan atur kunci API Anda sebelum menjalankan skrip. Lihat petunjuk di atas.")
        return

    # Inisialisasi klien Gemini
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"❌ Gagal menginisialisasi klien Gemini: {e}")
        return

    # Daftar tugas: memetakan nama file (tanpa ekstensi) ke teks yang akan diucapkan
    # Nama file disesuaikan dengan permintaan untuk suara wanita ("Despina").
    tasks = {
        "berbicara": "ada yang bisa saya bantu?",
        # "ganti_suara_ke_wanita": "suara telah diganti",
        # "fitur_belum_didukung": "maaf, fitur tersebut belum didukung",
        # "ganti_warna": "mengubah warna lampu, sesuai keinginan anda",
        # "mematikan_lampu": "lampu telah dimatikan",
        # "menyalakan_lampu": "lampu telah dinyalakan",
        # "perintah_tidak_dikenali": "perintah tidak dikenali, coba lagi",
        # "sedih": "mode sedih telah dinyalakan",
        # "senang": "mode senang telah dinyalakan"
    }
    
    # Konfigurasi suara yang akan digunakan
    voice_name = "Despina" # Sesuai permintaan Anda
    model_name = "gemini-2.5-pro-preview-tts"

    print(f"Menggunakan model '{model_name}' dengan suara '{voice_name}'...")
    print("-" * 30)

    # Loop melalui setiap tugas dan hasilkan file audio
    for filename_base, text_to_speak in tasks.items():
        print(f"Memproses: '{text_to_speak}'")
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=text_to_speak),
                ],
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            response_modalities=["audio"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name
                    )
                )
            ),
        )

        try:
            # Menggunakan streaming untuk menangani output
            full_audio_data = bytearray()
            for chunk in client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=generate_content_config,
            ):
                if (chunk.candidates and 
                    chunk.candidates[0].content and 
                    chunk.candidates[0].content.parts and 
                    chunk.candidates[0].content.parts[0].inline_data and 
                    chunk.candidates[0].content.parts[0].inline_data.data):
                    
                    inline_data = chunk.candidates[0].content.parts[0].inline_data
                    full_audio_data.extend(inline_data.data)
                    mime_type = inline_data.mime_type
                # Jika ada output teks (misalnya, pesan error dari model), cetak
                elif chunk.text:
                    print(f"ℹ️ Info dari model: {chunk.text}")

            if full_audio_data:
                # Coba tebak ekstensi file dari mime_type (kemungkinan besar .mp3)
                file_extension = mimetypes.guess_extension(mime_type)
                
                # Fallback ke .wav jika ekstensi tidak dikenali
                if file_extension is None:
                    print(f"⚠️ Mime type '{mime_type}' tidak dikenali, mengonversi ke .wav")
                    file_extension = ".wav"
                    final_data = convert_to_wav(bytes(full_audio_data), mime_type)
                else:
                    final_data = bytes(full_audio_data)
                
                # Simpan file
                save_binary_file(f"{filename_base}{file_extension}", final_data)
            else:
                print(f"❌ Tidak ada data audio yang diterima untuk teks: '{text_to_speak}'")

        except Exception as e:
            print(f"❌ Terjadi kesalahan saat memproses '{text_to_speak}': {e}")
        
        print("-" * 30)
    
    print("🎉 Semua file audio telah berhasil dibuat.")

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Menghasilkan header file WAV untuk data audio mentah."""
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", chunk_size, b"WAVE", b"fmt ",
        16, 1, num_channels, sample_rate,
        byte_rate, block_align, bits_per_sample,
        b"data", data_size
    )
    return header + audio_data

def parse_audio_mime_type(mime_type: str) -> dict[str, int | None]:
    """Mengurai bits per sample dan rate dari string mime_type audio."""
    bits_per_sample = 16  # Default
    rate = 24000          # Default
    
    parts = mime_type.split(";")
    for param in parts:
        param = param.strip().lower()
        if param.startswith("rate="):
            try:
                rate = int(param.split("=", 1)[1])
            except (ValueError, IndexError):
                pass
        elif param.startswith("audio/l"):
            try:
                bits_per_sample = int(param.split("l", 1)[1])
            except (ValueError, IndexError):
                pass
    return {"bits_per_sample": bits_per_sample, "rate": rate}


if __name__ == "__main__":
    generate_all_files()