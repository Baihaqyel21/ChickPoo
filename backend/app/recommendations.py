CLASS_LABELS = {
    "cocci": "Coccidiosis",
    "healthy": "Sehat",
    "ncd": "Newcastle Disease",
    "salmo": "Salmonella",
}

CLASS_DESCRIPTIONS = {
    "cocci": "Pola feses mengarah ke dugaan coccidiosis.",
    "healthy": "Pola feses tampak mendekati kondisi normal.",
    "ncd": "Pola feses mengarah ke dugaan Newcastle Disease.",
    "salmo": "Pola feses mengarah ke dugaan Salmonella.",
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
    "Pastikan foto yang dimasukkan benar-benar kotoran ayam, bukan kandang, pakan, tanah, tangan, atau objek lain.",
    "Bila ayam menunjukkan gejala sakit, tetap lakukan observasi dan konsultasi meskipun hasil foto belum kuat.",
]

INVALID_INPUT_RECOMMENDATIONS = [
    "Ambil ulang foto dengan objek utama berupa kotoran ayam, bukan tangan, pakan, tubuh ayam, atau benda lain.",
    "Dekatkan kamera sampai kotoran memenuhi sebagian besar frame agar warna dan teksturnya terlihat jelas.",
    "Gunakan pencahayaan cukup dan hindari latar putih polos yang membuat objek sulit dibedakan dari foto non-feses.",
    "Jika memang foto sudah berisi feses ayam, ulangi dari sudut yang lebih dekat dan pastikan fokus kamera terkunci pada objek.",
]

DISCLAIMER = (
    "ChickPoo adalah alat pemeriksaan awal berbasis gambar, bukan diagnosis veteriner final. "
    "Keputusan penanganan penyakit tetap perlu mempertimbangkan gejala klinis, kondisi kandang, "
    "dan konfirmasi petugas kesehatan hewan."
)


def confidence_band(confidence: float) -> dict:
    if confidence >= 0.80:
        return {
            "level": "tinggi",
            "message": "Tingkat keyakinan pada hasil utama cukup tinggi.",
        }
    if confidence >= 0.60:
        return {
            "level": "sedang",
            "message": "Hasil dapat menjadi indikasi awal, tetapi foto dan kondisi ayam tetap perlu diperiksa ulang.",
        }
    return {
        "level": "rendah",
        "message": "Tingkat keyakinan masih rendah. Foto mungkin kurang jelas atau objek belum terlihat baik.",
    }


def build_recommendation(label: str, confidence: float) -> dict:
    band = confidence_band(confidence)

    if confidence < 0.60:
        headline = "Hasil belum cukup kuat untuk dijadikan acuan."
        actions = UNCERTAIN_RECOMMENDATIONS
    else:
        headline = CLASS_DESCRIPTIONS[label]
        actions = RECOMMENDATIONS[label]

    return {
        "headline": headline,
        "confidence_level": band["level"],
        "confidence_message": band["message"],
        "actions": actions,
        "photo_notes": [],
        "disclaimer": DISCLAIMER,
    }


def build_invalid_input_recommendation(object_validation: dict) -> dict:
    probability = object_validation.get("percentage_chicken_feces", 0)
    threshold = object_validation.get("threshold", 0.69) * 100
    notes = [
        (
            "Penyaring awal memberi skor kesesuaian "
            f"{probability:.2f}%, sedangkan ambang lolos adalah {threshold:.0f}%."
        )
    ]
    return {
        "headline": "Foto belum cukup sesuai untuk dibaca.",
        "confidence_level": "perlu foto ulang",
        "confidence_message": "Pemeriksaan penyakit ditahan agar hasil tidak berasal dari objek yang salah.",
        "actions": INVALID_INPUT_RECOMMENDATIONS,
        "photo_notes": notes,
        "disclaimer": DISCLAIMER,
    }
