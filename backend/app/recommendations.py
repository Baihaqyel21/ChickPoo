CLASS_LABELS = {
    "cocci": "Coccidiosis",
    "healthy": "Sehat",
    "ncd": "Newcastle Disease",
    "salmo": "Salmonella",
}

CLASS_DESCRIPTIONS = {
    "cocci": "Foto menunjukkan pola yang paling mirip dengan kelas coccidiosis pada dataset.",
    "healthy": "Foto menunjukkan pola yang paling mirip dengan kelas feses ayam sehat pada dataset.",
    "ncd": "Foto menunjukkan pola yang paling mirip dengan kelas Newcastle Disease pada dataset.",
    "salmo": "Foto menunjukkan pola yang paling mirip dengan kelas Salmonella pada dataset.",
}

RECOMMENDATIONS = {
    "healthy": [
        "Lanjutkan pemantauan kondisi ayam secara rutin, terutama perubahan nafsu makan, aktivitas, dan warna feses.",
        "Jaga litter tetap kering, bersihkan area pakan dan minum, serta kurangi penumpukan kotoran di kandang.",
        "Catat hasil pemantauan harian agar perubahan kecil dapat terlihat lebih cepat.",
        "Ambil foto ulang bila kondisi feses berubah atau muncul gejala lain pada ayam.",
    ],
    "cocci": [
        "Pisahkan ayam yang dicurigai sakit dari kelompok utama untuk mengurangi risiko penularan.",
        "Bersihkan litter atau alas kandang yang lembap, karena lingkungan basah dapat memperburuk risiko coccidiosis.",
        "Pantau konsumsi pakan dan air minum, kondisi tubuh, serta adanya feses berdarah atau berlendir.",
        "Konsultasikan ke petugas kesehatan hewan atau dokter hewan untuk memastikan tindakan dan terapi yang tepat.",
    ],
    "salmo": [
        "Perkuat sanitasi kandang, tempat pakan, tempat minum, dan alat pembersih yang kontak dengan kotoran.",
        "Batasi kontak ayam yang dicurigai sakit dengan ayam lain, terutama bila ada gejala lemah atau diare.",
        "Gunakan sarung tangan atau alat pelindung saat membersihkan kotoran karena Salmonella juga berisiko bagi manusia.",
        "Hubungi petugas kesehatan hewan bila gejala berlanjut atau muncul kasus serupa pada beberapa ayam.",
    ],
    "ncd": [
        "Segera isolasi ayam yang dicurigai sakit karena Newcastle Disease dapat menyebar cepat di kandang.",
        "Hindari perpindahan ayam, peralatan, atau alas kandang dari area terduga sakit ke area ayam sehat.",
        "Perhatikan gejala tambahan seperti gangguan pernapasan, lemas, penurunan makan, atau perubahan perilaku.",
        "Segera hubungi petugas kesehatan hewan untuk konfirmasi dan arahan penanganan lanjutan.",
    ],
}

UNCERTAIN_RECOMMENDATIONS = [
    "Ambil foto ulang dengan objek feses terlihat dekat dan memenuhi sebagian besar frame.",
    "Gunakan pencahayaan cukup, hindari bayangan keras, foto buram, atau flash yang memantul berlebihan.",
    "Pastikan foto yang dimasukkan benar-benar foto feses ayam, bukan kandang, pakan, tanah, tangan, atau objek lain.",
    "Bila ayam menunjukkan gejala sakit, tetap lakukan observasi dan konsultasi meskipun model belum yakin.",
]

INVALID_INPUT_RECOMMENDATIONS = [
    "Ambil ulang foto dengan objek utama berupa kotoran atau feses ayam, bukan tangan, pakan, tubuh ayam, atau benda lain.",
    "Dekatkan kamera sampai feses memenuhi sebagian besar frame agar sistem dapat membaca tekstur dan warna dengan lebih baik.",
    "Gunakan pencahayaan cukup dan hindari latar putih polos yang membuat objek sulit dibedakan dari foto non-feses.",
    "Jika memang foto sudah berisi feses ayam, ulangi dari sudut yang lebih dekat dan pastikan fokus kamera terkunci pada objek.",
]

DISCLAIMER = (
    "ChickPoo adalah alat screening awal berbasis gambar, bukan diagnosis veteriner final. "
    "Keputusan penanganan penyakit tetap perlu mempertimbangkan gejala klinis, kondisi kandang, "
    "dan konfirmasi petugas kesehatan hewan."
)


def confidence_band(confidence: float) -> dict:
    if confidence >= 0.80:
        return {
            "level": "tinggi",
            "message": "Model cukup yakin dengan prediksi utama.",
        }
    if confidence >= 0.60:
        return {
            "level": "sedang",
            "message": "Model memberi indikasi awal, tetapi foto sebaiknya tetap diperiksa ulang.",
        }
    return {
        "level": "rendah",
        "message": "Model belum cukup yakin. Foto mungkin kurang jelas atau objek tidak sesuai.",
    }


def build_recommendation(label: str, confidence: float, quality_flags: list[dict]) -> dict:
    band = confidence_band(confidence)
    quality_messages = [flag["message"] for flag in quality_flags]

    if confidence < 0.60:
        headline = "Model belum yakin terhadap foto ini."
        actions = UNCERTAIN_RECOMMENDATIONS
    else:
        headline = CLASS_DESCRIPTIONS[label]
        actions = RECOMMENDATIONS[label]

    return {
        "headline": headline,
        "confidence_level": band["level"],
        "confidence_message": band["message"],
        "actions": actions,
        "photo_notes": quality_messages,
        "disclaimer": DISCLAIMER,
    }


def build_invalid_input_recommendation(quality_flags: list[dict], relevance_flags: list[dict]) -> dict:
    notes = [flag["message"] for flag in relevance_flags] + [flag["message"] for flag in quality_flags]
    return {
        "headline": "Foto belum dikenali sebagai feses ayam.",
        "confidence_level": "perlu foto ulang",
        "confidence_message": "Sistem menahan prediksi penyakit karena objek foto terindikasi tidak sesuai.",
        "actions": INVALID_INPUT_RECOMMENDATIONS,
        "photo_notes": notes,
        "disclaimer": DISCLAIMER,
    }
