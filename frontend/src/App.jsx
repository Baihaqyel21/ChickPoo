import {
  AlertTriangle,
  ArrowRight,
  BarChart3,
  Camera,
  CheckCircle2,
  ChevronRight,
  FileText,
  Gauge,
  ImagePlus,
  LoaderCircle,
  RotateCcw,
  ScanSearch,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Upload,
  Wheat,
  X,
} from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

const navItems = [
  { id: 'home', label: 'Beranda' },
  { id: 'scanner', label: 'Scanner' },
  { id: 'workflow', label: 'Cara Kerja' },
  { id: 'model', label: 'Model' },
];

const sampleImages = [
  { src: '/samples/healthy.jpg', label: 'Sehat', filename: 'healthy-sample.jpg' },
  { src: '/samples/cocci.jpg', label: 'Coccidiosis', filename: 'cocci-sample.jpg' },
  { src: '/samples/ncd.jpg', label: 'Newcastle', filename: 'ncd-sample.jpg' },
  { src: '/samples/salmo.jpg', label: 'Salmonella', filename: 'salmo-sample.jpg' },
];

const modelMetrics = [
  { label: 'Validasi Objek', value: '99.54%', note: 'akurasi pada test set validasi objek' },
  { label: 'Presisi Gate', value: '99.90%', note: 'ketepatan saat menerima feses ayam' },
  { label: 'Accuracy', value: '98.34%', note: 'akurasi model klasifikasi penyakit' },
  { label: 'Macro F1', value: '96.91%', note: 'metrik utama untuk kelas tidak seimbang' },
];

const classInfo = [
  { label: 'Sehat', key: 'healthy', tone: 'good', text: 'Pola visual mendekati kondisi normal.' },
  { label: 'Coccidiosis', key: 'cocci', tone: 'warning', text: 'Perlu pemantauan karena dapat mengganggu produktivitas.' },
  { label: 'Newcastle Disease', key: 'ncd', tone: 'danger', text: 'Kelas berisiko tinggi dan perlu tindak lanjut cepat.' },
  { label: 'Salmonella', key: 'salmo', tone: 'warning', text: 'Berkaitan dengan potensi gangguan pencernaan dan sanitasi.' },
];

const statusIcon = {
  accepted: CheckCircle2,
  review: AlertTriangle,
  needs_retake: ShieldAlert,
  invalid_input: ShieldAlert,
};

function formatPercent(value) {
  return `${Number(value || 0).toFixed(2)}%`;
}

function objectValidationPercent(result) {
  return result?.object_validation?.percentage_chicken_feces ?? 0;
}

function isLikelyFeces(result) {
  return result?.object_validation?.status === 'likely_feces';
}

function photoSuitabilityPercent(result) {
  if (isLikelyFeces(result)) {
    return result?.object_validation?.visual_support?.support_score ?? objectValidationPercent(result);
  }
  return objectValidationPercent(result);
}

function resultTone(result) {
  if (!result) return 'idle';
  if (result.status === 'invalid_input') return 'danger';
  if (result.predicted_class === 'healthy') return 'good';
  if (result.predicted_class === 'ncd') return 'danger';
  return 'warning';
}

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    if (!cameraActive || !videoRef.current || !streamRef.current) return;
    videoRef.current.srcObject = streamRef.current;
    videoRef.current.play().catch(() => {
      setCameraError('Kamera belum bisa menampilkan preview. Tutup lalu buka lagi.');
    });
  }, [cameraActive]);

  function scrollToSection(sectionId) {
    document.getElementById(sectionId)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function updateSelectedFile(file) {
    if (!file) return;
    setError('');
    setResult(null);
    setSelectedFile(file);
    setPreviewUrl((previousUrl) => {
      if (previousUrl) URL.revokeObjectURL(previousUrl);
      return URL.createObjectURL(file);
    });
  }

  function handleFileInput(event) {
    updateSelectedFile(event.target.files?.[0]);
    event.target.value = '';
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragActive(false);
    const file = event.dataTransfer.files?.[0];
    if (file?.type?.startsWith('image/')) {
      updateSelectedFile(file);
    } else {
      setError('Gunakan file gambar seperti JPG, PNG, atau WEBP.');
    }
  }

  function handleDrag(event, active) {
    event.preventDefault();
    setDragActive(active);
  }

  async function useSampleImage(item) {
    const response = await fetch(item.src);
    const blob = await response.blob();
    updateSelectedFile(new File([blob], item.filename, { type: blob.type || 'image/jpeg' }));
    scrollToSection('scanner');
  }

  async function startCamera() {
    setCameraError('');
    setError('');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' } },
        audio: false,
      });
      streamRef.current = stream;
      setCameraActive(true);
      scrollToSection('scanner');
    } catch {
      setCameraError('Kamera tidak bisa diakses. Periksa izin kamera di browser.');
    }
  }

  function stopCamera() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setCameraActive(false);
  }

  function capturePhoto() {
    const video = videoRef.current;
    if (!video || !video.videoWidth) {
      setCameraError('Kamera belum siap. Tunggu sebentar lalu coba lagi.');
      return;
    }
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) return;
      updateSelectedFile(new File([blob], `chickpoo-camera-${Date.now()}.jpg`, { type: 'image/jpeg' }));
      stopCamera();
    }, 'image/jpeg', 0.92);
  }

  async function submitPrediction() {
    if (!selectedFile) {
      setError('Pilih atau ambil foto terlebih dahulu.');
      return;
    }
    setIsLoading(true);
    setError('');
    setResult(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || 'Analisis belum bisa diproses.');
      }
      setResult(payload);
    } catch (requestError) {
      setError(requestError.message || 'Layanan belum siap. Coba beberapa saat lagi.');
    } finally {
      setIsLoading(false);
    }
  }

  function resetInput() {
    setSelectedFile(null);
    setResult(null);
    setError('');
    setCameraError('');
    setPreviewUrl((previousUrl) => {
      if (previousUrl) URL.revokeObjectURL(previousUrl);
      return '';
    });
  }

  return (
    <main className="app-shell">
      <AmbientBackground />
      <TopNav onNavigate={scrollToSection} />
      <HeroSection onStart={() => scrollToSection('scanner')} onLearn={() => scrollToSection('workflow')} />
      <ScannerSection
        selectedFile={selectedFile}
        previewUrl={previewUrl}
        cameraActive={cameraActive}
        cameraError={cameraError}
        dragActive={dragActive}
        isLoading={isLoading}
        result={result}
        error={error}
        fileInputRef={fileInputRef}
        videoRef={videoRef}
        handleFileInput={handleFileInput}
        handleDrop={handleDrop}
        handleDrag={handleDrag}
        startCamera={startCamera}
        stopCamera={stopCamera}
        capturePhoto={capturePhoto}
        submitPrediction={submitPrediction}
        resetInput={resetInput}
        useSampleImage={useSampleImage}
      />
      <WorkflowSection onStart={() => scrollToSection('scanner')} />
      <ModelSection onStart={() => scrollToSection('scanner')} />
      <AboutSection onStart={() => scrollToSection('scanner')} />
    </main>
  );
}

function AmbientBackground() {
  return <div className="ambient" aria-hidden="true" />;
}

function TopNav({ onNavigate }) {
  return (
    <header className="topbar">
      <button className="brand" type="button" onClick={() => onNavigate('home')} title="Kembali ke beranda">
        <span className="brand-mark">
          <img src="/assets/poop.png" alt="" />
        </span>
        <span>
          <strong>ChickPoo</strong>
          <small>Pemeriksaan awal</small>
        </span>
      </button>
      <nav className="nav-links" aria-label="Navigasi utama">
        {navItems.map((item) => (
          <button key={item.id} type="button" onClick={() => onNavigate(item.id)}>
            {item.label}
          </button>
        ))}
      </nav>
      <button className="nav-cta" type="button" onClick={() => onNavigate('scanner')}>
        <ScanSearch size={18} />
        <span>Mulai Scan</span>
      </button>
    </header>
  );
}

function HeroSection({ onStart, onLearn }) {
  return (
    <section className="hero-section" id="home">
      <div className="hero-copy">
        <p className="eyebrow">Peternakan modern, keputusan lebih cepat</p>
        <h1>Screening Kesehatan Ayam dari Foto Feses</h1>
        <p>
          ChickPoo membantu peternak membaca sinyal awal kesehatan ayam dari foto feses,
          lalu menyajikan peluang kondisi dan rekomendasi tindakan dengan bahasa yang mudah dipahami.
        </p>
        <div className="hero-actions">
          <button className="primary-action" type="button" onClick={onStart}>
            <ScanSearch size={20} />
            <span>Mulai Scan</span>
          </button>
          <button className="secondary-action" type="button" onClick={onLearn}>
            <span>Lihat Cara Kerja</span>
            <ArrowRight size={18} />
          </button>
        </div>
        <div className="hero-proof" aria-label="Ringkasan alur ChickPoo">
          <span>Upload foto</span>
          <span>Screening objek</span>
          <span>Rekomendasi tindakan</span>
        </div>
      </div>

      <aside className="hero-visual" aria-label="Visual peternakan ChickPoo">
        <div className="farm-card">
          <img src="/assets/background-landingpage.jpg" alt="Peternakan ayam modern" />
          <div className="farm-card-glow" aria-hidden="true" />
          <div className="farm-caption">
            <span>ChickPoo</span>
            <strong>Dari kandang, keputusan awal lebih cepat.</strong>
          </div>
        </div>
        <div className="floating-card floating-card-top">
          <ShieldCheck size={18} />
          <span>Foto dicek lebih dulu sebelum hasil penyakit tampil.</span>
        </div>
        <div className="floating-card floating-card-bottom">
          <Sparkles size={18} />
          <span>Hasil dibuat ringkas, transparan, dan mudah dijelaskan.</span>
        </div>
      </aside>
    </section>
  );
}

function ScannerSection({
  selectedFile,
  previewUrl,
  cameraActive,
  cameraError,
  dragActive,
  isLoading,
  result,
  error,
  fileInputRef,
  videoRef,
  handleFileInput,
  handleDrop,
  handleDrag,
  startCamera,
  stopCamera,
  capturePhoto,
  submitPrediction,
  resetInput,
  useSampleImage,
}) {
  return (
    <section className="scanner-section" id="scanner">
      <SectionIntro
        kicker="Scanner"
        title="Unggah foto feses ayam untuk memulai screening awal."
        text="Flow dibuat sederhana: unggah, lihat preview, analisis, lalu baca hasil dan rekomendasi."
      />

      <div className="scanner-grid">
        <div className="scanner-card glass-panel">
          <StepIndicator result={result} isLoading={isLoading} selectedFile={selectedFile} />
          <div
            className={`drop-zone ${dragActive ? 'dragging' : ''} ${previewUrl || cameraActive ? 'has-preview' : ''}`}
            onDragOver={(event) => handleDrag(event, true)}
            onDragLeave={(event) => handleDrag(event, false)}
            onDrop={handleDrop}
          >
            {cameraActive ? (
              <video ref={videoRef} className="preview-media" muted playsInline />
            ) : previewUrl ? (
              <img src={previewUrl} alt="Preview foto feses ayam" className="preview-media" />
            ) : (
              <div className="drop-empty">
                <span className="drop-icon">
                  <ImagePlus size={42} />
                </span>
                <h3>Tarik gambar ke sini</h3>
                <p>Atau pilih foto dari perangkat. Gunakan gambar yang dekat, terang, dan fokus.</p>
              </div>
            )}
          </div>

          <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileInput} hidden />

          <div className="scanner-actions">
            <button className="secondary-action" type="button" onClick={() => fileInputRef.current?.click()}>
              <Upload size={18} />
              <span>Pilih Gambar</span>
            </button>
            {cameraActive ? (
              <>
                <button className="secondary-action accent" type="button" onClick={capturePhoto}>
                  <Camera size={18} />
                  <span>Ambil Foto</span>
                </button>
                <button className="icon-action" type="button" onClick={stopCamera} title="Tutup kamera">
                  <X size={18} />
                </button>
              </>
            ) : (
              <button className="secondary-action" type="button" onClick={startCamera}>
                <Camera size={18} />
                <span>Kamera</span>
              </button>
            )}
          </div>

          <button className="scan-action" type="button" onClick={submitPrediction} disabled={isLoading || !selectedFile}>
            {isLoading ? <LoaderCircle className="spin" size={20} /> : <ScanSearch size={20} />}
            <span>{isLoading ? 'Gambar sedang diperiksa...' : 'Analisis Sekarang'}</span>
          </button>

          {(selectedFile || result) && (
            <button className="reset-link" type="button" onClick={resetInput}>
              <RotateCcw size={16} />
              <span>Upload Ulang</span>
            </button>
          )}

          {cameraError && <p className="surface-alert">{cameraError}</p>}
          {error && <p className="surface-alert danger">{error}</p>}
        </div>

        <ResultPanel result={result} isLoading={isLoading} resetInput={resetInput} />
      </div>

      <SampleStrip useSampleImage={useSampleImage} />
    </section>
  );
}

function StepIndicator({ result, isLoading, selectedFile }) {
  const currentStep = useMemo(() => {
    if (result?.status === 'invalid_input') return 1;
    if (result) return 3;
    if (isLoading) return 2;
    if (selectedFile) return 1;
    return 0;
  }, [result, isLoading, selectedFile]);

  const steps = [
    { number: '01', label: 'Validasi objek' },
    { number: '02', label: 'Prediksi penyakit' },
    { number: '03', label: 'Rekomendasi awal' },
  ];

  return (
    <div className="stepper" aria-label="Tahapan screening ChickPoo">
      {steps.map((step, index) => (
        <div className={`step-pill ${currentStep >= index + 1 ? 'active' : ''}`} key={step.number}>
          <span>{step.number}</span>
          <strong>{step.label}</strong>
        </div>
      ))}
    </div>
  );
}

function ResultPanel({ result, isLoading, resetInput }) {
  const StatusIcon = result ? statusIcon[result.status] || AlertTriangle : Gauge;
  const tone = resultTone(result);

  if (isLoading) {
    return (
      <aside className="result-card glass-panel loading-state">
        <div className="loading-orbit">
          <LoaderCircle className="spin" size={42} />
        </div>
        <h2>Gambar sedang diperiksa...</h2>
        <p>Sistem sedang memastikan foto sesuai sebelum membaca kemungkinan kondisi kesehatan.</p>
      </aside>
    );
  }

  if (!result) {
    return (
      <aside className="result-card glass-panel empty-state">
        <div className="empty-orb">
          <Gauge size={42} />
        </div>
        <p className="eyebrow">Belum Ada Hasil</p>
        <h2>Hasil screening akan muncul di sini.</h2>
        <p>Unggah foto feses ayam yang jelas untuk melihat dugaan kondisi dan rekomendasi awal.</p>
      </aside>
    );
  }

  const invalidInput = result.status === 'invalid_input';
  const confidence = invalidInput ? photoSuitabilityPercent(result) : result.confidence_percentage;

  return (
    <aside className={`result-card glass-panel ${tone}`}>
      <div className="result-status">
        <span>
          <StatusIcon size={19} />
          {invalidInput ? 'Objek tidak sesuai' : 'Feses ayam terdeteksi'}
        </span>
        <strong>{result.recommendation.confidence_level}</strong>
      </div>

      <div className="result-hero">
        <div className="score-ring" style={{ '--score': `${Math.min(Number(confidence || 0), 100) * 3.6}deg` }}>
          <span>{formatPercent(confidence)}</span>
        </div>
        <div>
          <p className="eyebrow">{invalidInput ? 'Status Foto' : 'Hasil Screening'}</p>
          <h2>{result.predicted_label}</h2>
          <p>{result.status_message}</p>
        </div>
      </div>

      {invalidInput ? (
        <InvalidInputBlock result={result} resetInput={resetInput} />
      ) : (
        <DiseaseResult result={result} />
      )}

      <RecommendationCard recommendation={result.recommendation} />
    </aside>
  );
}

function InvalidInputBlock({ result, resetInput }) {
  return (
    <div className="invalid-block">
      <p>
        Gambar yang diunggah belum terdeteksi sebagai feses ayam. Silakan gunakan foto feses ayam
        yang lebih jelas agar analisis dapat dilanjutkan.
      </p>
      <div className="mini-meter">
        <span>Skor kesesuaian foto</span>
        <strong>{formatPercent(photoSuitabilityPercent(result))}</strong>
      </div>
      <div className="progress-track">
        <span style={{ width: `${Math.max(photoSuitabilityPercent(result), 1)}%` }} />
      </div>
      <button className="secondary-action accent" type="button" onClick={resetInput}>
        <RotateCcw size={17} />
        <span>Upload Ulang</span>
      </button>
    </div>
  );
}

function DiseaseResult({ result }) {
  return (
    <div className="disease-panel">
      {isLikelyFeces(result) && (
        <div className="soft-warning">
          Foto terindikasi sesuai, tetapi hasil tetap sebaiknya dibaca sebagai indikasi awal.
        </div>
      )}
      <div className="probability-list">
        {result.probabilities.map((item) => (
          <div className={`probability-row ${item.key}`} key={item.key}>
            <div className="probability-meta">
              <span>{item.label}</span>
              <strong>{formatPercent(item.percentage)}</strong>
            </div>
            <div className="progress-track" aria-hidden="true">
              <span style={{ width: `${Math.max(item.percentage, 1)}%` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function RecommendationCard({ recommendation }) {
  return (
    <article className="recommendation-card">
      <p className="eyebrow">Rekomendasi Awal</p>
      <h3>{recommendation.headline}</h3>
      <p>{recommendation.confidence_message}</p>
      <ol>
        {recommendation.actions.map((action) => (
          <li key={action}>{action}</li>
        ))}
      </ol>
      <small>{recommendation.disclaimer}</small>
    </article>
  );
}

function SampleStrip({ useSampleImage }) {
  return (
    <div className="sample-strip">
      <div>
        <p className="eyebrow">Contoh Cepat</p>
        <h3>Coba contoh cepat</h3>
      </div>
      <div className="sample-actions">
        {sampleImages.map((item) => (
          <button type="button" key={item.label} onClick={() => useSampleImage(item)} title={`Pakai contoh ${item.label}`}>
            <img src={item.src} alt={`Contoh ${item.label}`} />
            <span>{item.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function WorkflowSection({ onStart }) {
  const steps = [
    ['01', 'Upload foto feses ayam', 'Ambil foto yang dekat, terang, dan fokus.'],
    ['02', 'Foto diperiksa', 'Gambar dicek agar objek yang salah tidak langsung dianalisis.'],
    ['03', 'Kondisi diprediksi', 'Jika foto sesuai, sistem membaca kemungkinan kondisi kesehatan.'],
    ['04', 'Rekomendasi muncul', 'Peternak mendapat langkah awal yang mudah diikuti.'],
  ];

  return (
    <section className="workflow-section" id="workflow">
      <SectionIntro
        kicker="Cara Kerja ChickPoo"
        title="Dibuat singkat agar mudah dipakai di kandang."
        text="ChickPoo membantu mempercepat keputusan awal tanpa membuat pengguna masuk ke alur yang rumit."
      />
      <div className="workflow-grid">
        {steps.map(([number, title, text]) => (
          <article className="workflow-card glass-panel" key={number}>
            <span>{number}</span>
            <h3>{title}</h3>
            <p>{text}</p>
          </article>
        ))}
      </div>
      <div className="why-card glass-panel">
        <Wheat size={30} />
        <div>
          <p className="eyebrow">Kenapa Penting?</p>
          <h3>Deteksi awal membantu peternak bergerak lebih cepat.</h3>
          <p>
            Penyakit yang terlambat dikenali dapat menyebar di kandang dan menambah kerugian.
            Screening awal memberi sinyal agar tindakan pencegahan bisa dimulai lebih cepat.
          </p>
        </div>
        <button className="primary-action" type="button" onClick={onStart}>
          <span>Mulai Scan</span>
          <ChevronRight size={18} />
        </button>
      </div>
    </section>
  );
}

function ModelSection({ onStart }) {
  return (
    <section className="model-section" id="model">
      <SectionIntro
        kicker="Model"
        title="Penjelasan teknis ditempatkan di sini."
        text="Bagian ini disiapkan untuk dosen dan presentasi kelas, tanpa membebani tampilan scanner."
      />

      <div className="metrics-grid">
        {modelMetrics.map((item) => (
          <article className="metric-card glass-panel" key={item.label}>
            <span>{item.label}</span>
            <strong>{item.value}</strong>
            <p>{item.note}</p>
          </article>
        ))}
      </div>

      <div className="model-layout">
        <div className="evidence-stack">
          <figure className="evidence-figure glass-panel">
            <img src="/assets/object-validation-confusion-matrix.png" alt="Confusion matrix model validasi objek ChickPoo" />
            <figcaption>Confusion matrix model validasi objek: feses ayam vs objek lain.</figcaption>
          </figure>
          <figure className="evidence-figure glass-panel">
            <img src="/assets/confusion-matrix.png" alt="Confusion matrix model klasifikasi penyakit ChickPoo" />
            <figcaption>Confusion matrix model klasifikasi penyakit pada test set.</figcaption>
          </figure>
        </div>

        <section className="model-explain glass-panel">
          <p className="eyebrow">Didukung Computer Vision</p>
          <h3>Model bertingkat untuk mengurangi prediksi asal.</h3>
          <div className="pipeline-steps">
            <article>
              <span>01</span>
              <strong>Validasi objek</strong>
              <p>Model pertama mengecek apakah gambar cukup sesuai sebagai feses ayam.</p>
            </article>
            <article>
              <span>02</span>
              <strong>Klasifikasi penyakit</strong>
              <p>Jika lolos, model kedua memprediksi healthy, coccidiosis, salmonella, atau Newcastle disease.</p>
            </article>
          </div>
          <p className="model-note">
            Model menganalisis pola visual seperti warna, tekstur, bentuk, dan konsistensi feses.
            Hasil tetap diposisikan sebagai screening awal, bukan diagnosis final.
          </p>
          <div className="class-list">
            {classInfo.map((item) => (
              <article className={`class-item ${item.tone}`} key={item.key}>
                <strong>{item.label}</strong>
                <p>{item.text}</p>
              </article>
            ))}
          </div>
          <button className="primary-action" type="button" onClick={onStart}>
            <ScanSearch size={18} />
            <span>Coba Scanner</span>
          </button>
        </section>
      </div>
    </section>
  );
}

function AboutSection({ onStart }) {
  return (
    <section className="about-section">
      <div className="closing-card glass-panel">
        <p className="eyebrow">Tentang ChickPoo</p>
        <h2>Alat screening awal untuk peternak kecil.</h2>
        <p>
          ChickPoo membantu membaca tanda awal dari foto feses, menyajikan tingkat keyakinan, dan
          memberi rekomendasi awal. Untuk keputusan pengobatan, tetap konsultasikan dengan petugas
          kesehatan hewan.
        </p>
        <button className="primary-action" type="button" onClick={onStart}>
          <span>Mulai Scan</span>
          <ArrowRight size={18} />
        </button>
      </div>
    </section>
  );
}

function SectionIntro({ kicker, title, text }) {
  return (
    <div className="section-intro">
      <p className="eyebrow">{kicker}</p>
      <h2>{title}</h2>
      <p>{text}</p>
    </div>
  );
}

export default App;
